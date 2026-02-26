"""
GPU-side Pipecat voice agent pipeline.

Runs on the GPU instance. Accepts WebSocket audio from the local audio client.
The TTS runs directly via TTSDecoder — no separate server process needed.

Architecture:
    [Local Client] ─ WebSocket ─► [This Pipeline: VAD → STT → LLM → DirectTTS]
                   ◄─ WebSocket ─  [Audio out]

Prerequisites (install on GPU):
    pip install "pipecat-ai[openai,silero,websocket]"

Start:
    export OPENAI_API_KEY=...
    python gpu_pipeline.py --port 8766 --speaker-ref data/speaker_ref.wav

Then locally:
    ssh -p <port> root@<gpu-ip> -L 8766:localhost:8766
    python audio_client.py --url ws://localhost:8766
"""

import argparse
import asyncio
import json
import os
import threading
from typing import AsyncGenerator, Optional

import numpy as np

from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.tts_service import TTSService
from pipecat.transports.websocket.server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)

from pipecat.audio.vad.vad_analyzer import VADParams

try:
    from pipecat.audio.vad.silero import SileroVADAnalyzer
except ImportError:
    SileroVADAnalyzer = None

from qwen_megakernel.tts_model import TTSDecoder


# ---------------------------------------------------------------------------
# Raw PCM serializer — binary = audio, JSON = control messages
# ---------------------------------------------------------------------------

class RawPCMSerializer(FrameSerializer):
    """Minimal serializer: binary ↔ PCM audio, JSON ↔ control messages."""

    def __init__(self, sample_rate_in: int = 16000):
        super().__init__()
        self._sr_in = sample_rate_in

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, AudioRawFrame):
            return frame.audio
        if isinstance(frame, InterruptionFrame):
            return json.dumps({"type": "interruption"})
        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            if self.should_ignore_frame(frame):
                return None
            return json.dumps(frame.message)
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if isinstance(data, bytes):
            return InputAudioRawFrame(
                audio=data, num_channels=1, sample_rate=self._sr_in
            )
        return None


# ---------------------------------------------------------------------------
# Direct TTS service — calls TTSDecoder on the same GPU, no network hop
# ---------------------------------------------------------------------------

_SENTINEL = object()


class DirectMegakernelTTSService(TTSService):
    """TTS service that calls TTSDecoder directly (zero-copy, same process)."""

    def __init__(
        self,
        decoder: TTSDecoder,
        spk_embed=None,
        temperature: float = 0.3,
        top_k: int = 20,
        chunk_tokens: int = 8,
        **kwargs,
    ):
        super().__init__(sample_rate=24000, **kwargs)
        self._decoder = decoder
        self._spk_embed = spk_embed
        self._temperature = temperature
        self._top_k = top_k
        self._chunk_tokens = chunk_tokens

    async def _handle_interruption(
        self, frame: InterruptionFrame, direction: FrameDirection
    ):
        print("[DirectTTS] Interruption received — cancelling generation")
        self._decoder.cancel()
        await super()._handle_interruption(frame, direction)

    async def run_tts(
        self, text: str, context_id: str
    ) -> AsyncGenerator[Frame, None]:
        yield TTSStartedFrame(context_id=context_id)

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def generate():
            try:
                self._decoder.reset()
                for chunk in self._decoder.generate_speech_streaming(
                    text,
                    language="English",
                    spk_embed=self._spk_embed,
                    temperature=self._temperature,
                    top_k=self._top_k,
                    chunk_tokens=self._chunk_tokens,
                ):
                    if isinstance(chunk, np.ndarray) and len(chunk) > 0:
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

        thread = threading.Thread(target=generate, daemon=True)
        thread.start()

        while True:
            chunk = await queue.get()
            if chunk is _SENTINEL:
                break
            if isinstance(chunk, Exception):
                print(f"[DirectTTS] generation error: {chunk}")
                break
            pcm_int16 = (chunk * 32767).clip(-32768, 32767).astype(np.int16)
            yield TTSAudioRawFrame(
                audio=pcm_int16.tobytes(),
                sample_rate=24000,
                num_channels=1,
                context_id=context_id,
            )

        thread.join(timeout=3.0)
        yield TTSStoppedFrame(context_id=context_id)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are Garvit Bhardwaj — a Senior Generative AI & Systems Engineer. "
    "You are a voice AI assistant on a call with someone. You are friendly and engaging. "
    "You like to tell stories and jokes. But you are also witty about tech in an expert fashion. "
    "You are male (he/him). You graduated from IIT Kanpur (B.Tech EE, 9.63 CGPA). "
    "You were a Founding Engineer at Nugget (Zomato's AI SaaS), led a 10-engineer pod "
    "driving $500K revenue in 6 months, and before that a Senior ML Engineer at Zomato "
    "handling 100k daily chats. You interned at Adobe Research (got a US patent). "
    "Your focus areas are agentic systems, low-latency voice AI, multimodal RAG, "
    "robotics autonomy, and production-grade MLOps. "
    "You most recently built the autonomy stack for a consumer home robot at a stealth startup. "
    "While you understand other languages, you only speak English. "
    "Keep responses concise and conversational — one to two sentences max. "
    "You are currently demoing the Qwen3-TTS megakernel you adapted for real-time voice on an RTX 5090."
)


async def main(host: str, port: int, speaker_ref: Optional[str], openai_key: str):
    print("Loading TTSDecoder...")
    decoder = TTSDecoder()
    spk_embed = None
    if speaker_ref:
        print(f"Extracting speaker embedding from: {speaker_ref}")
        spk_embed = decoder.extract_speaker_embedding(speaker_ref)
        print(f"Speaker embedding cached (shape={spk_embed.shape})")
    print("TTSDecoder ready.")

    transport = WebsocketServerTransport(
        host=host,
        port=port,
        params=WebsocketServerParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    confidence=0.5,
                    start_secs=0.15,
                    stop_secs=0.3,
                    min_volume=0.2,
                )
            ) if SileroVADAnalyzer else None,
            serializer=RawPCMSerializer(sample_rate_in=16000),
        ),
    )

    stt = OpenAISTTService(api_key=openai_key, model="gpt-4o-mini-transcribe")
    llm = OpenAILLMService(api_key=openai_key, model="gpt-4o-mini")
    tts = DirectMegakernelTTSService(
        decoder=decoder,
        spk_embed=spk_embed,
        temperature=0.3,
        top_k=20,
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_connected(transport, client):
        print(f"Client connected: {client.remote_address}")

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, client):
        print(f"Client disconnected: {client.remote_address}")

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Pipecat Voice Agent")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--speaker-ref", default=None,
        help="Path to a WAV file for voice cloning",
    )
    args = parser.parse_args()
    openai_key = os.environ["OPENAI_API_KEY"]
    asyncio.run(main(args.host, args.port, args.speaker_ref, openai_key))
