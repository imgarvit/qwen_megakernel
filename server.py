"""
WebSocket TTS server — runs on the GPU instance.

Protocol:
  Client sends JSON:  {"text": "Hello world", "language": "English", "temperature": 0.3, "top_k": 15}
  Server streams:     binary PCM frames (float32, 24kHz, mono)
  Server sends JSON:  {"done": true, "duration_s": 3.04, "steps": 38, "rtf": 0.82}
  Client sends JSON:  {"cancel": true}   — cancels current generation

Start:
  python server.py --host 0.0.0.0 --port 8765
  python server.py --host 0.0.0.0 --port 8765 --speaker-ref /path/to/voice.wav
"""

import argparse
import asyncio
import json
import struct
import time

import numpy as np
import websockets

from qwen_megakernel.tts_model import TTSDecoder


class TTSServer:
    def __init__(self, speaker_ref: str | None = None):
        print("Loading TTSDecoder...")
        self._decoder = TTSDecoder()
        self._default_speaker_ref = speaker_ref
        if speaker_ref:
            print(f"Default speaker reference: {speaker_ref}")
        print("TTSDecoder ready.")

    async def _listen_for_cancel(self, websocket):
        """Background task: watch for cancel messages during generation."""
        try:
            async for raw_msg in websocket:
                try:
                    msg = json.loads(raw_msg)
                except (json.JSONDecodeError, TypeError):
                    continue
                if msg.get("cancel"):
                    self._decoder.cancel()
                    return
                if msg.get("text"):
                    self._pending_msg = msg
                    self._decoder.cancel()
                    return
        except websockets.exceptions.ConnectionClosed:
            self._decoder.cancel()

    async def handle(self, websocket):
        remote = websocket.remote_address
        print(f"[{remote}] connected")
        self._pending_msg = None
        try:
            async for raw_msg in websocket:
                msg = self._pending_msg or None
                self._pending_msg = None
                if msg is None:
                    try:
                        msg = json.loads(raw_msg)
                    except (json.JSONDecodeError, TypeError):
                        await websocket.send(json.dumps({"error": "invalid JSON"}))
                        continue

                if msg.get("cancel"):
                    continue

                text = msg.get("text", "")
                if not text:
                    await websocket.send(json.dumps({"error": "empty text"}))
                    continue

                language = msg.get("language", "English")
                temperature = float(msg.get("temperature", 0.3))
                top_k = int(msg.get("top_k", 15))
                chunk_tokens = int(msg.get("chunk_tokens", 8))
                speaker_ref = msg.get("speaker_ref", self._default_speaker_ref)

                print(f"[{remote}] TTS: {text[:60]}...")
                t0 = time.perf_counter()
                ttfc = None
                total_samples = 0
                steps = 0
                cancelled = False

                self._decoder.reset()
                for audio_chunk in self._decoder.generate_speech_streaming(
                    text,
                    language=language,
                    speaker_ref=speaker_ref,
                    temperature=temperature,
                    top_k=top_k,
                    chunk_tokens=chunk_tokens,
                ):
                    if isinstance(audio_chunk, np.ndarray) and len(audio_chunk) > 0:
                        if ttfc is None:
                            ttfc = time.perf_counter() - t0
                        pcm = audio_chunk.astype(np.float32).tobytes()
                        await websocket.send(pcm)
                        total_samples += len(audio_chunk)
                    steps += 1

                elapsed = time.perf_counter() - t0
                duration = total_samples / 24000 if total_samples else 0
                rtf = elapsed / duration if duration > 0 else 0

                done_msg = {
                    "done": True,
                    "duration_s": round(duration, 3),
                    "generation_s": round(elapsed, 3),
                    "ttfc_ms": round((ttfc or 0) * 1000, 1),
                    "rtf": round(rtf, 3),
                    "samples": total_samples,
                }
                await websocket.send(json.dumps(done_msg))
                print(f"[{remote}] done: {duration:.2f}s audio in {elapsed:.2f}s (RTF={rtf:.3f}, TTFC={((ttfc or 0)*1000):.0f}ms)")

        except websockets.exceptions.ConnectionClosed:
            self._decoder.cancel()
            print(f"[{remote}] disconnected")


async def main(host: str, port: int, speaker_ref: str | None = None):
    server = TTSServer(speaker_ref=speaker_ref)
    print(f"Starting WebSocket server on ws://{host}:{port}")
    async with websockets.serve(
        server.handle,
        host,
        port,
        max_size=2**20,
        ping_interval=30,
        ping_timeout=60,
    ):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Megakernel TTS WebSocket server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--speaker-ref", default=None,
                        help="Path to a WAV file used as the default speaker voice")
    args = parser.parse_args()
    asyncio.run(main(args.host, args.port, speaker_ref=args.speaker_ref))
