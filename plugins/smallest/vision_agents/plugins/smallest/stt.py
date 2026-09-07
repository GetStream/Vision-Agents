"""Smallest AI Speech-to-Text via WebSocket streaming (Pulse).

Docs:
    - Quickstart: https://docs.smallest.ai/waves/documentation/speech-to-text-pulse/realtime-web-socket/quickstart
    - Response format: https://docs.smallest.ai/waves/documentation/speech-to-text-pulse/realtime-web-socket/response-format

Pulse is the only model available on this streaming endpoint (Pulse Pro is
pre-recorded only and returns 400 here).
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Optional
from urllib.parse import urlencode

import aiohttp
from getstream.video.rtc.track_util import PcmData
from vision_agents.core import stt
from vision_agents.core.edge.types import Participant
from vision_agents.core.stt import TranscriptResponse

logger = logging.getLogger(__name__)

WS_URL = "wss://api.smallest.ai/waves/v1/stt/live"
MODEL = "pulse"


class STT(stt.STT):
    """Smallest AI Pulse streaming Speech-to-Text.

    Pulse doesn't emit VAD/turn-boundary events on this WebSocket, so turn
    detection is left to an external ``turn_detection`` component.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        language: str = "en",
        sample_rate: int = 16000,
        word_timestamps: bool = False,
    ) -> None:
        """Initialize Smallest AI STT.

        Args:
            api_key: Smallest AI API key. Falls back to ``SMALLEST_API_KEY`` env var.
            language: Language code (e.g. ``en``, ``hi``). Pulse supports 38 languages.
            sample_rate: Input sample rate in Hz. Defaults to 16000.
            word_timestamps: Request per-word timestamps in the response.
        """
        super().__init__(provider_name="smallest")

        self._api_key = api_key or os.environ.get("SMALLEST_API_KEY")
        if not self._api_key:
            raise ValueError(
                "SMALLEST_API_KEY env var or api_key parameter required for Smallest AI STT"
            )

        self.model = MODEL
        self.language = language
        self.sample_rate = sample_rate
        self.word_timestamps = word_timestamps

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._receive_task: Optional[asyncio.Task[None]] = None
        self._connection_ready = asyncio.Event()
        self._current_participant: Optional[Participant] = None
        self._audio_start_time: Optional[float] = None

    async def start(self) -> None:
        """Open the Smallest AI WebSocket and start the receive loop."""
        await super().start()

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "X-Source": "getstream",
        }
        self._session = aiohttp.ClientSession()
        self._ws = await self._session.ws_connect(self._build_ws_url(), headers=headers)

        self._receive_task = asyncio.create_task(self._receive_loop())
        self._connection_ready.set()
        self._on_connected()

    async def process_audio(
        self,
        pcm_data: PcmData,
        participant: Participant,
    ) -> None:
        """Send a PCM audio chunk to Smallest AI as a binary WebSocket frame."""
        if self.closed:
            logger.warning("Smallest AI STT is closed, ignoring audio")
            return

        await self._connection_ready.wait()

        if self._ws is None or self._ws.closed:
            logger.warning("Smallest AI STT WebSocket not open, dropping audio")
            return

        resampled = pcm_data.resample(self.sample_rate, 1)
        audio_bytes = resampled.samples.tobytes()

        self._current_participant = participant
        if self._audio_start_time is None:
            self._audio_start_time = time.perf_counter()

        await self._ws.send_bytes(audio_bytes)

    async def close(self) -> None:
        """Send close_stream, close the WebSocket, and clean up."""
        await super().close()

        if self._ws is not None and not self._ws.closed:
            try:
                await self._ws.send_str(json.dumps({"type": "close_stream"}))
            except (aiohttp.ClientError, ConnectionError):
                logger.debug("Could not send close_stream to Smallest AI")
            await self._ws.close()
        self._ws = None

        if self._receive_task is not None:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

        self._connection_ready.clear()
        self._on_disconnected()
        self._audio_start_time = None

    def _build_ws_url(self) -> str:
        params: dict[str, str | int] = {
            "model": self.model,
            "language": self.language,
            "encoding": "linear16",
            "sample_rate": self.sample_rate,
            "word_timestamps": "true" if self.word_timestamps else "false",
        }
        return f"{WS_URL}?{urlencode(params)}"

    async def _receive_loop(self) -> None:
        ws = self._ws
        if ws is None:
            return
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        parsed = json.loads(msg.data)
                    except json.JSONDecodeError:
                        logger.warning(
                            "Smallest AI STT sent non-JSON text: %s", msg.data
                        )
                        continue
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("Smallest AI STT message: %s", parsed)
                    self._handle_message(parsed)
                elif msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                    aiohttp.WSMsgType.ERROR,
                ):
                    break
        except asyncio.CancelledError:
            raise
        except aiohttp.ClientError:
            logger.exception("Smallest AI STT receive loop error")

        if not self.closed:
            self._emit_error_event(
                ConnectionError("Smallest AI STT WebSocket closed unexpectedly"),
                context="smallest_ws_closed",
            )

    def _handle_message(self, data: dict[str, Any]) -> None:
        """Dispatch a parsed Smallest AI WebSocket transcript message."""
        if "error" in data or data.get("status") == "error":
            err_msg = (
                data.get("error") or data.get("message") or "Smallest AI STT error"
            )
            self._emit_error_event(
                Exception(str(err_msg)), context="smallest_streaming"
            )
            return

        if data.get("is_last"):
            return

        transcript_text = data.get("transcript") or ""
        if not transcript_text:
            return

        participant = self._current_participant
        if participant is None:
            logger.warning("Smallest AI transcript received but no participant set")
            return

        processing_time_ms: Optional[float] = None
        if self._audio_start_time is not None:
            processing_time_ms = (time.perf_counter() - self._audio_start_time) * 1000

        is_final = bool(data.get("is_final"))
        response = TranscriptResponse(
            language=data.get("language") or self.language,
            model_name=self.model,
            processing_time_ms=processing_time_ms,
        )

        self._emit_transcript_event(
            transcript_text,
            participant,
            response,
            mode="final" if is_final else "replacement",
        )
        if is_final:
            self._audio_start_time = None
