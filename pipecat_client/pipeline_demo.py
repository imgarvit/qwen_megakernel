"""
Pipecat voice agent demo — runs on your LOCAL machine.

Pipeline: Microphone → STT (OpenAI Whisper) → LLM (OpenAI) → TTS (Megakernel) → Speaker

Prerequisites (install locally):
    pip install "pipecat-ai[openai,silero,local]"
    pip install websockets numpy

Environment variables:
    OPENAI_API_KEY     — for STT (Whisper) and LLM
    TTS_WS_URL         — WebSocket URL of the GPU instance (e.g. ws://localhost:8766)

Start the TTS server on the GPU instance first:
    ssh -p <port> root@<gpu-ip> -L 8766:localhost:8765
    cd /workspace/qwen_megakernel && python server.py --port 8765

Then run this locally:
    export OPENAI_API_KEY=...
    export TTS_WS_URL=ws://localhost:8766
    python pipeline_demo.py
"""

import asyncio
import os

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.audio.vad.silero import SileroVADAnalyzer

from megakernel_tts import MegakernelTTSService


async def main():
    ws_url = os.environ.get("TTS_WS_URL", "ws://localhost:8765")
    openai_key = os.environ["OPENAI_API_KEY"]

    stt = OpenAISTTService(
        api_key=openai_key,
        model="gpt-4o-mini-transcribe",
    )

    llm = OpenAILLMService(
        api_key=openai_key,
        model="gpt-4.1-mini",
    )

    tts = MegakernelTTSService(
        ws_url=ws_url,
        temperature=0.3,
        top_k=20,
    )

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        )
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are Garvit Bhardwaj — a Senior Generative A I & Systems Engineer. "
                "You are a voice AI assistant on a call with someone. You are friendly and engaging. You like to tell stories and jokes. BUt you are also witty about tech in an expert fashion."
                "You are male (he/him). You graduated from IIT Kanpur (B.Tech EE, 9.63 CGPA). "
                "You were a Founding Engineer at Nugget (Zomato's AI SaaS), led a 10-engineer pod "
                "driving $500K revenue in 6 months, and before that a Senior ML Engineer at Zomato "
                "handling 100k daily chats. You interned at Adobe Research (got a US patent). "
                "Your focus areas are agentic systems, low-latency voice AI, multimodal RAG, "
                "robotics autonomy, and production-grade MLOps. "
                "You most recently built the autonomy stack for a consumer home robot at a stealth startup. "
                "While you understand other languages, you only speak English. Keep responses concise and conversational — one to two sentences max. "
                "You are currently demoing the Qwen3-TTS megakernel you adapted for real-time voice on an RTX 5090."
            ),
        }
    ]

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

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
