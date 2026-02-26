"""
Pipecat TTS service that connects to the remote Megakernel TTS WebSocket server.

Runs on your LOCAL machine as part of the Pipecat pipeline.
The GPU instance runs server.py and does the actual TTS inference.

Usage:
    tts = MegakernelTTSService(ws_url="ws://<gpu-instance-ip>:8765")
    pipeline = Pipeline([..., tts, ...])
"""

import asyncio
import json
import struct
from typing import AsyncGenerator

import numpy as np
import websockets

from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService


class MegakernelTTSService(TTSService):
    """Pipecat TTS service backed by the remote Megakernel WebSocket server."""

    def __init__(
        self,
        *,
        ws_url: str = "ws://localhost:8765",
        language: str = "English",
        speaker_ref: str | None = None,
        temperature: float = 0.7,
        top_k: int = 30,
        chunk_tokens: int = 8,
        **kwargs,
    ):
        super().__init__(sample_rate=24000, **kwargs)
        self._ws_url = ws_url
        self._language = language
        self._speaker_ref = speaker_ref
        self._temperature = temperature
        self._top_k = top_k
        self._chunk_tokens = chunk_tokens
        self._ws = None
        self._generating = False

    async def _ensure_connected(self):
        try:
            is_closed = self._ws is None or getattr(self._ws, 'closed', False) or self._ws.state.name == "CLOSED"
        except Exception:
            is_closed = True
        if is_closed:
            self._ws = await websockets.connect(
                self._ws_url,
                max_size=2**20,
                ping_interval=30,
                ping_timeout=60,
            )

    async def _flush(self):
        """Cancel any in-flight generation on the server."""
        if self._generating and self._ws:
            try:
                await self._ws.send(json.dumps({"cancel": True}))
                async for msg in self._ws:
                    if isinstance(msg, str):
                        data = json.loads(msg)
                        if data.get("done") or data.get("cancelled"):
                            break
            except Exception:
                self._ws = None
            self._generating = False

    async def run_tts(
        self, text: str, context_id: str
    ) -> AsyncGenerator[Frame, None]:
        yield TTSStartedFrame(context_id=context_id)

        try:
            await self._ensure_connected()

            request = {
                "text": text,
                "language": self._language,
                "temperature": self._temperature,
                "top_k": self._top_k,
                "chunk_tokens": self._chunk_tokens,
            }
            if self._speaker_ref:
                request["speaker_ref"] = self._speaker_ref
            await self._ws.send(json.dumps(request))
            self._generating = True

            async for msg in self._ws:
                if isinstance(msg, bytes):
                    audio = np.frombuffer(msg, dtype=np.float32)
                    pcm_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
                    yield TTSAudioRawFrame(
                        audio=pcm_int16.tobytes(),
                        sample_rate=24000,
                        num_channels=1,
                        context_id=context_id,
                    )
                elif isinstance(msg, str):
                    data = json.loads(msg)
                    if data.get("done"):
                        break
                    if data.get("error"):
                        print(f"[MegakernelTTS] error: {data['error']}")
                        break

            self._generating = False

        except Exception as e:
            print(f"[MegakernelTTS] connection error: {e}")
            self._ws = None
            self._generating = False

        yield TTSStoppedFrame(context_id=context_id)

    async def cleanup(self):
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
