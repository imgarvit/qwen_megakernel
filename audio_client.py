"""
Local audio client — streams mic/speaker audio to the GPU Pipecat instance.

Captures microphone audio, sends it over WebSocket to the GPU pipeline,
receives synthesized speech audio back, and plays it through the speakers.

With --record, saves the full conversation (both sides) to a WAV file.

Usage:
    # 1. Start the GPU pipeline (on the GPU instance):
    #    OPENAI_API_KEY=... python gpu_pipeline.py --port 8766 --speaker-ref data/speaker_ref.wav

    # 2. Open SSH tunnel (locally):
    #    ssh -p <ssh-port> root@<gpu-ip> -L 8766:localhost:8766

    # 3. Run this client (locally):
    python audio_client.py
    python audio_client.py --url ws://localhost:8766 --record demo.wav
"""

import argparse
import asyncio
import time
import wave

import numpy as np
import pyaudio
import websockets
import websockets.exceptions

SAMPLE_RATE_IN = 16000
SAMPLE_RATE_OUT = 24000
CHANNELS = 1
CHUNK_MS = 20
CHUNK_IN = SAMPLE_RATE_IN * CHUNK_MS // 1000
CHUNK_OUT = SAMPLE_RATE_OUT * CHUNK_MS // 1000


def resample_16k_to_24k(pcm_bytes: bytes) -> bytes:
    """Resample 16kHz int16 PCM to 24kHz using linear interpolation."""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    n_out = int(len(samples) * SAMPLE_RATE_OUT / SAMPLE_RATE_IN)
    indices = np.linspace(0, len(samples) - 1, n_out)
    resampled = np.interp(indices, np.arange(len(samples)), samples)
    return resampled.astype(np.int16).tobytes()


async def main(url: str, record_path: str | None = None):
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

    t0 = time.monotonic()
    mic_entries: list[tuple[float, bytes]] = []
    bot_entries: list[tuple[float, bytes]] = []
    bot_playback_time = 0.0
    last_bot_arrival = 0.0

    print(f"Connecting to {url} ...")
    async with websockets.connect(
        url, max_size=2**20, ping_interval=30, ping_timeout=60
    ) as ws:
        msg = "Connected! Speak into your microphone. Ctrl+C to quit."
        if record_path:
            msg += f"\nRecording to: {record_path}"
        print(msg + "\n")

        async def send_mic():
            while True:
                try:
                    data = in_stream.read(CHUNK_IN, exception_on_overflow=False)
                    await ws.send(data)
                    if record_path:
                        t = time.monotonic() - t0
                        resampled = resample_16k_to_24k(data)
                        mic_entries.append((t, resampled))
                except Exception as e:
                    print(f"[send] {e}")
                    break
                await asyncio.sleep(0.001)

        async def recv_speaker():
            nonlocal bot_playback_time, last_bot_arrival
            try:
                async for msg in ws:
                    if isinstance(msg, bytes):
                        out_stream.write(msg)
                        if record_path:
                            now = time.monotonic() - t0
                            chunk_dur = (len(msg) / 2) / SAMPLE_RATE_OUT
                            gap = now - last_bot_arrival
                            if last_bot_arrival == 0 or gap > 0.3:
                                bot_playback_time = now
                            bot_entries.append((bot_playback_time, msg))
                            bot_playback_time += chunk_dur
                            last_bot_arrival = now
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

    if record_path and (mic_entries or bot_entries):
        print(f"Saving recording to {record_path} ...")
        total_dur = time.monotonic() - t0
        total_samples = int(total_dur * SAMPLE_RATE_OUT) + SAMPLE_RATE_OUT
        mic_track = np.zeros(total_samples, dtype=np.float32)
        bot_track = np.zeros(total_samples, dtype=np.float32)

        for t, data in mic_entries:
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            pos = int(t * SAMPLE_RATE_OUT)
            end = min(pos + len(samples), total_samples)
            if pos < total_samples:
                mic_track[pos:end] = samples[: end - pos]

        for t, data in bot_entries:
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            pos = int(t * SAMPLE_RATE_OUT)
            end = min(pos + len(samples), total_samples)
            if pos < total_samples:
                bot_track[pos:end] = samples[: end - pos]

        mixed = (mic_track * 0.5 + bot_track * 0.7).clip(-32768, 32767).astype(np.int16)
        final_len = max(
            max((int(t * SAMPLE_RATE_OUT) + len(d) // 2) for t, d in mic_entries) if mic_entries else 0,
            max((int(t * SAMPLE_RATE_OUT) + len(d) // 2) for t, d in bot_entries) if bot_entries else 0,
        )
        mixed = mixed[:final_len]

        with wave.open(record_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE_OUT)
            wf.writeframes(mixed.tobytes())

        duration = final_len / SAMPLE_RATE_OUT
        print(f"Saved {duration:.1f}s recording to {record_path}")

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
    parser.add_argument(
        "--record",
        default=None,
        metavar="FILE",
        help="Record conversation to WAV file (e.g. --record demo.wav)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args.url, args.record))
    except KeyboardInterrupt:
        print("\nBye!")
