"""
WebSocket TTS server — runs on the GPU instance.

Protocol:
  Client sends JSON:  {"text": "Hello world", "language": "English"}
  Server streams:     binary PCM frames (float32, 24kHz, mono)
  Server sends JSON:  {"done": true, ...metrics...}

  Cancellation (client may send at any time during generation):
    Client sends JSON:  {"cancel": true}
    Server stops decode immediately, replies:
      {"cancelled": true, "steps": N}

Start:
  python server.py --host 0.0.0.0 --port 8765
  python server.py --host 0.0.0.0 --port 8765 --speaker-ref /path/to/voice.wav
"""

import argparse
import asyncio
import json
import threading
import time

import numpy as np
import websockets

from qwen_megakernel.tts_model import TTSDecoder

_SENTINEL = object()


class TTSServer:
    def __init__(self, speaker_ref: str | None = None):
        print("Loading TTSDecoder...")
        self._decoder = TTSDecoder()
        self._cached_spk_embed = None
        if speaker_ref:
            print(f"Extracting speaker embedding from: {speaker_ref}")
            self._cached_spk_embed = self._decoder.extract_speaker_embedding(speaker_ref)
            print(f"Speaker embedding cached (shape={self._cached_spk_embed.shape})")
        print("TTSDecoder ready.")

    def _run_generation_thread(self, queue, loop, text, language, temperature, top_k, chunk_tokens):
        """Synchronous generation in a background thread. Pushes chunks into an asyncio queue."""
        try:
            for audio_chunk in self._decoder.generate_speech_streaming(
                text,
                language=language,
                spk_embed=self._cached_spk_embed,
                temperature=temperature,
                top_k=top_k,
                chunk_tokens=chunk_tokens,
            ):
                if isinstance(audio_chunk, np.ndarray) and len(audio_chunk) > 0:
                    loop.call_soon_threadsafe(queue.put_nowait, audio_chunk)
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, e)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    async def _listen_for_cancel(self, websocket):
        """Concurrently watch for cancel or new request while generation runs."""
        try:
            async for raw_msg in websocket:
                try:
                    msg = json.loads(raw_msg)
                except (json.JSONDecodeError, TypeError):
                    continue
                if msg.get("cancel"):
                    self._decoder.cancel()
                    return "cancelled"
                if msg.get("text"):
                    self._pending_msg = msg
                    self._decoder.cancel()
                    return "new_request"
        except websockets.exceptions.ConnectionClosed:
            self._decoder.cancel()
            return "disconnected"
        return None

    async def handle(self, websocket):
        remote = websocket.remote_address
        print(f"[{remote}] connected")
        self._pending_msg = None
        loop = asyncio.get_event_loop()

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
                temperature = float(msg.get("temperature", 0.9))
                top_k = int(msg.get("top_k", 50))
                chunk_tokens = int(msg.get("chunk_tokens", 8))

                print(f"[{remote}] TTS: {text[:60]}...")
                t0 = time.perf_counter()
                ttfc = None
                total_samples = 0
                steps = 0

                self._decoder.reset()
                queue = asyncio.Queue()

                gen_thread = threading.Thread(
                    target=self._run_generation_thread,
                    args=(queue, loop, text, language, temperature, top_k, chunk_tokens),
                    daemon=True,
                )
                gen_thread.start()

                cancel_task = asyncio.create_task(self._listen_for_cancel(websocket))

                cancelled = False
                try:
                    while True:
                        chunk = await queue.get()
                        if chunk is _SENTINEL:
                            break
                        if isinstance(chunk, Exception):
                            print(f"[{remote}] generation error: {chunk}")
                            break
                        if ttfc is None:
                            ttfc = time.perf_counter() - t0
                        pcm = chunk.astype(np.float32).tobytes()
                        await websocket.send(pcm)
                        total_samples += len(chunk)
                        steps += 1
                finally:
                    cancel_task.cancel()
                    try:
                        cancel_result = await cancel_task
                    except asyncio.CancelledError:
                        cancel_result = None

                    gen_thread.join(timeout=2.0)

                    cancelled = self._decoder._cancelled

                elapsed = time.perf_counter() - t0
                duration = total_samples / 24000 if total_samples else 0
                rtf = elapsed / duration if duration > 0 else 0

                if cancelled:
                    done_msg = {"cancelled": True, "steps": steps}
                    print(f"[{remote}] cancelled after {steps} steps ({elapsed:.2f}s)")
                else:
                    done_msg = {
                        "done": True,
                        "duration_s": round(duration, 3),
                        "generation_s": round(elapsed, 3),
                        "ttfc_ms": round((ttfc or 0) * 1000, 1),
                        "rtf": round(rtf, 3),
                        "samples": total_samples,
                    }
                    print(f"[{remote}] done: {duration:.2f}s audio in {elapsed:.2f}s (RTF={rtf:.3f}, TTFC={((ttfc or 0)*1000):.0f}ms)")

                await websocket.send(json.dumps(done_msg))

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
        await asyncio.Future()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Megakernel TTS WebSocket server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--speaker-ref", default=None,
                        help="Path to a WAV file used as the default speaker voice")
    args = parser.parse_args()
    asyncio.run(main(args.host, args.port, speaker_ref=args.speaker_ref))
