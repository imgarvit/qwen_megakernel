"""
Standalone test client for the Megakernel TTS WebSocket server.
No Pipecat dependency — just connects, sends text, saves audio.

Usage:
    python test_client.py --url ws://<gpu-ip>:8765 --text "Hello world"
"""

import argparse
import asyncio
import json
import time

import numpy as np
import soundfile as sf
import websockets


async def test_tts(url: str, text: str, output: str):
    print(f"Connecting to {url}...")
    async with websockets.connect(url, max_size=2**20) as ws:
        request = {
            "text": text,
            "language": "English",
            "temperature": 0.9,
            "top_k": 50,
        }
        print(f"Sending: {text}")
        t0 = time.perf_counter()
        await ws.send(json.dumps(request))

        audio_chunks = []
        first_chunk_time = None

        async for msg in ws:
            if isinstance(msg, bytes):
                chunk = np.frombuffer(msg, dtype=np.float32)
                audio_chunks.append(chunk)
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter() - t0
                    print(f"  TTFC: {first_chunk_time*1000:.0f}ms")
            elif isinstance(msg, str):
                data = json.loads(msg)
                if data.get("done"):
                    elapsed = time.perf_counter() - t0
                    print(f"  Server stats: {data}")
                    print(f"  Client elapsed: {elapsed:.2f}s")
                    break

        if audio_chunks:
            audio = np.concatenate(audio_chunks)
            sf.write(output, audio, 24000)
            print(f"  Saved: {output} ({len(audio)/24000:.2f}s)")
        else:
            print("  No audio received!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8765")
    parser.add_argument("--text", default="Hello! This is a test of the megakernel text to speech system.")
    parser.add_argument("--output", default="test_output.wav")
    args = parser.parse_args()
    asyncio.run(test_tts(args.url, args.text, args.output))
