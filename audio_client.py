"""
Local audio client — streams mic/speaker audio to the GPU Pipecat instance.

Captures microphone audio, sends it over WebSocket to the GPU pipeline,
receives synthesized speech audio back, and plays it through the speakers.

Usage:
    # 1. Start the GPU pipeline (on the GPU instance):
    #    OPENAI_API_KEY=... python gpu_pipeline.py --port 8766 --speaker-ref data/speaker_ref.wav

    # 2. Open SSH tunnel (locally):
    #    ssh -p <ssh-port> root@<gpu-ip> -L 8766:localhost:8766

    # 3. Run this client (locally):
    python audio_client.py
    python audio_client.py --url ws://localhost:8766
"""

import argparse
import asyncio

import pyaudio
import websockets
import websockets.exceptions

SAMPLE_RATE_IN = 16000
SAMPLE_RATE_OUT = 24000
CHANNELS = 1
CHUNK_MS = 20
CHUNK_IN = SAMPLE_RATE_IN * CHUNK_MS // 1000
CHUNK_OUT = SAMPLE_RATE_OUT * CHUNK_MS // 1000


async def main(url: str):
    p = pyaudio.PyAudio()

    in_stream = p.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE_IN,
        input=True,
        frames_per_buffer=CHUNK_IN,
    )

    out_stream = p.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE_OUT,
        output=True,
        frames_per_buffer=CHUNK_OUT,
    )

    print(f"Connecting to {url} ...")
    async with websockets.connect(
        url, max_size=2**20, ping_interval=30, ping_timeout=60
    ) as ws:
        print("Connected! Speak into your microphone. Ctrl+C to quit.\n")

        async def send_mic():
            while True:
                try:
                    data = in_stream.read(CHUNK_IN, exception_on_overflow=False)
                    await ws.send(data)
                except Exception as e:
                    print(f"[send] {e}")
                    break
                await asyncio.sleep(0.001)

        async def recv_speaker():
            try:
                async for msg in ws:
                    if isinstance(msg, bytes):
                        out_stream.write(msg)
            except websockets.exceptions.ConnectionClosed:
                print("Server closed the connection.")
            except Exception as e:
                print(f"[recv] {e}")

        try:
            await asyncio.gather(send_mic(), recv_speaker())
        except asyncio.CancelledError:
            pass

    in_stream.stop_stream()
    in_stream.close()
    out_stream.stop_stream()
    out_stream.close()
    p.terminate()
    print("Disconnected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Local audio client for GPU Pipecat voice agent"
    )
    parser.add_argument(
        "--url",
        default="ws://localhost:8766",
        help="WebSocket URL of the GPU pipeline (default: ws://localhost:8766)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args.url))
    except KeyboardInterrupt:
        print("\nBye!")
