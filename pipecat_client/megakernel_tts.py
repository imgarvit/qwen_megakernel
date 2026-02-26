"""
Pipecat TTS service that connects to the remote Megakernel TTS WebSocket server.

Runs on your LOCAL machine as part of the Pipecat pipeline.
The GPU instance runs server.py and does the actual TTS inference.

Interruption flow:
  1. VAD detects user speaking → Pipecat sends InterruptionFrame
  2. _handle_interruption sets _interrupt_event
  3. run_tts sees the event, sends {"cancel": true} to GPU, drains until ack
  4. GPU stops decode loop immediately, replies {"cancelled": true}
  5. TTS is ready for the next sentence — WebSocket stays alive

Usage:
    tts = MegakernelTTSService(ws_url="ws://<gpu-instance-ip>:8765")
    pipeline = Pipeline([..., tts, ...])
"""

import asyncio
import json
from typing import AsyncGenerator

import numpy as np
import websockets
import websockets.exceptions

from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import TTSService

DRAIN_TIMEOUT = 3.0


class MegakernelTTSService(TTSService):
    """Pipecat TTS service backed by the remote Megakernel WebSocket server."""

    def __init__(
        self,
        *,
        ws_url: str = "ws://localhost:8765",
        language: str = "English",
        temperature: float = 0.9,
        top_k: int = 50,
        chunk_tokens: int = 8,
        **kwargs,
    ):
        super().__init__(sample_rate=24000, **kwargs)
        self._ws_url = ws_url
        self._language = language
        self._temperature = temperature
        self._top_k = top_k
        self._chunk_tokens = chunk_tokens
        self._ws = None
        self._generating = False
        self._interrupt_event = asyncio.Event()

    async def _ensure_connected(self):
        try:
            is_closed = (
                self._ws is None
                or getattr(self._ws, "closed", False)
                or self._ws.state.name == "CLOSED"
            )
        except Exception:
            is_closed = True
        if is_closed:
            if self._ws is not None:
                try:
                    await self._ws.close()
                except Exception:
                    pass
            self._ws = await websockets.connect(
                self._ws_url,
                max_size=2**20,
                ping_interval=30,
                ping_timeout=60,
                open_timeout=5,
            )

    async def _send_cancel_and_drain(self):
        """Send cancel to the GPU and drain until ack. Keeps WS alive."""
        if not self._ws:
            self._generating = False
            return
        try:
            await self._ws.send(json.dumps({"cancel": True}))
            while True:
                msg = await asyncio.wait_for(self._ws.recv(), timeout=DRAIN_TIMEOUT)
                if isinstance(msg, str):
                    data = json.loads(msg)
                    if data.get("done") or data.get("cancelled"):
                        break
        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosedOK,
                websockets.exceptions.ConnectionClosedError):
            pass
        except Exception:
            pass
        finally:
            self._generating = False

    async def _close_ws(self):
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def _handle_interruption(
        self, frame: InterruptionFrame, direction: FrameDirection
    ):
        """Called by Pipecat when VAD detects the user speaking mid-utterance."""
        self._interrupt_event.set()
        await super()._handle_interruption(frame, direction)

    async def run_tts(
        self, text: str, context_id: str
    ) -> AsyncGenerator[Frame, None]:
        self._interrupt_event.clear()
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
            await self._ws.send(json.dumps(request))
            self._generating = True

            while True:
                if self._interrupt_event.is_set():
                    await self._send_cancel_and_drain()
                    break

                recv_task = asyncio.ensure_future(self._ws.recv())
                interrupt_task = asyncio.ensure_future(self._interrupt_event.wait())

                done, pending = await asyncio.wait(
                    {recv_task, interrupt_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass

                if interrupt_task in done:
                    recv_task.cancel()
                    try:
                        await recv_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    await self._send_cancel_and_drain()
                    break

                msg = recv_task.result()
                if isinstance(msg, bytes):
                    audio = np.frombuffer(msg, dtype=np.float32)
                    pcm_int16 = (
                        (audio * 32767).clip(-32768, 32767).astype(np.int16)
                    )
                    yield TTSAudioRawFrame(
                        audio=pcm_int16.tobytes(),
                        sample_rate=24000,
                        num_channels=1,
                        context_id=context_id,
                    )
                elif isinstance(msg, str):
                    data = json.loads(msg)
                    if data.get("done") or data.get("cancelled"):
                        self._generating = False
                        break
                    if data.get("error"):
                        print(f"[MegakernelTTS] error: {data['error']}")
                        self._generating = False
                        break

        except asyncio.CancelledError:
            if self._generating:
                await self._send_cancel_and_drain()
            raise
        except websockets.exceptions.ConnectionClosedOK:
            self._ws = None
            self._generating = False
        except Exception as e:
            print(f"[MegakernelTTS] connection error: {e}")
            self._ws = None
            self._generating = False

        yield TTSStoppedFrame(context_id=context_id)

    async def cleanup(self):
        await self._close_ws()
