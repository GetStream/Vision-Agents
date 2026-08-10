"""Telnyx Speech-to-Text via WebSocket streaming.

Docs: https://developers.telnyx.com/api/call-control/speech-to-text

Audio is sent as raw binary frames in the encoding named by ``input_format``.
The default is ``linear16``, for which Telnyx requires an explicit
``sample_rate``. Transcripts come back as
``{"transcript": str, "confidence": float | None, "is_final": bool}``.

The query parameter for partial transcripts is ``interim_results``.
``partial_results`` is accepted by the endpoint but ignored, which silently
yields finals only.

``interim_results`` itself is honoured per engine rather than per endpoint.
Measured against the live API with the same audio: ``Speechmatics`` and
``Soniox`` stream partials, while ``Telnyx`` and ``Deepgram`` accept the
parameter and return finals only.
"""

import asyncio
import json
import logging
import os
import time
from typing import Optional
from urllib.parse import urlencode

import aiohttp
from getstream.video.rtc.track_util import PcmData
from vision_agents.core import stt
from vision_agents.core.edge.types import Participant
from vision_agents.core.stt import TranscriptResponse

logger = logging.getLogger(__name__)

WS_STT_URL = "wss://api.telnyx.com/v2/speech-to-text/transcription"


class STT(stt.STT):
    """Telnyx streaming Speech-to-Text.

    Uses aiohttp for a fully async WebSocket connection to the Telnyx streaming
    endpoint. Audio is resampled to the configured rate and sent as raw
    ``linear16`` binary frames.

    Telnyx does not send VAD signals on this endpoint, so turn detection is
    left to the agent's own turn detection.

    Examples:

        from vision_agents.plugins import telnyx
        stt = telnyx.STT(sample_rate=8000)
    """

    turn_detection: bool = False

    def __init__(
        self,
        api_key: Optional[str] = None,
        transcription_engine: str = "Telnyx",
        language: str = "en",
        sample_rate: int = 16000,
        interim_results: bool = False,
        model: str = "",
    ) -> None:
        """Initialize Telnyx STT.

        Args:
            api_key: Telnyx API key. Falls back to the ``TELNYX_API_KEY`` env var.
            transcription_engine: Engine to transcribe with, for example
                ``Telnyx``, ``Deepgram`` or ``Speechmatics``. The catalogue is
                served by Telnyx and grows over time, so the value is not
                validated locally; an unknown engine is rejected by the server
                with an explicit error.
            language: Language code, for example ``en``.
            sample_rate: Rate in Hz that audio is resampled to before being
                sent. Use 8000 to pass telephony audio from
                :class:`TelnyxMediaStream` through without upsampling.
            interim_results: Emit partial transcripts as they are refined.
                Honoured per engine, not per endpoint: ``Speechmatics`` and
                ``Soniox`` stream partials, while the default ``Telnyx`` engine
                accepts the parameter and returns finals only. Defaults to
                ``False`` so the default configuration does not advertise
                partials it will never emit.
            model: Optional engine-specific model id.
        """
        super().__init__(provider_name="telnyx")

        self._api_key = api_key or os.environ.get("TELNYX_API_KEY")
        if not self._api_key:
            raise ValueError(
                "TELNYX_API_KEY env var or api_key parameter required for Telnyx STT"
            )

        self.transcription_engine = transcription_engine
        self.language = language
        self.sample_rate = sample_rate
        self.interim_results = interim_results
        self.model = model

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._receive_task: Optional[asyncio.Task[None]] = None
        self._connection_ready = asyncio.Event()
        self._current_participant: Optional[Participant] = None
        self._audio_start_time: Optional[float] = None
        self._error_reported = False

    async def start(self) -> None:
        """Open the Telnyx WebSocket and start the receive loop."""
        await super().start()

        # aiohttp does not attach an Origin header to the handshake. Clients
        # that do have to suppress it, because the Telnyx edge rejects a
        # WebSocket handshake that carries one.
        self._session = aiohttp.ClientSession()
        try:
            self._ws = await self._session.ws_connect(
                self._build_ws_url(),
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        except BaseException:
            await self._session.close()
            self._session = None
            raise

        self._receive_task = asyncio.create_task(self._receive_loop())
        self._connection_ready.set()
        self._on_connected()

    async def process_audio(
        self,
        pcm_data: PcmData,
        participant: Participant,
    ) -> None:
        """Resample a PCM chunk to the configured rate and send it to Telnyx."""
        if self.closed:
            logger.warning("Telnyx STT is closed, ignoring audio")
            return

        if not self.started:
            logger.warning("Telnyx STT is not started, dropping audio")
            return

        await self._connection_ready.wait()

        if self._ws is None or self._ws.closed:
            logger.warning("Telnyx STT WebSocket not open, dropping audio")
            return

        resampled = pcm_data.resample(self.sample_rate, 1)

        self._current_participant = participant
        if self._audio_start_time is None:
            self._audio_start_time = time.perf_counter()

        try:
            await self._ws.send_bytes(resampled.samples.tobytes())
        except (aiohttp.ClientError, ConnectionError) as exc:
            self._emit_error_event(exc, context="telnyx_send_audio")

    async def close(self) -> None:
        """Close the WebSocket and clean up."""
        await super().close()

        if self._ws is not None and not self._ws.closed:
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
        params: dict[str, str] = {
            "transcription_engine": self.transcription_engine,
            "input_format": "linear16",
            "sample_rate": str(self.sample_rate),
            "language": self.language,
            "interim_results": "true" if self.interim_results else "false",
        }
        if self.model:
            params["model"] = self.model
        return f"{WS_STT_URL}?{urlencode(params)}"

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
                        logger.warning("Telnyx STT sent non-JSON text: %s", msg.data)
                        continue
                    if not isinstance(parsed, dict):
                        logger.warning("Telnyx STT sent unexpected payload: %r", parsed)
                        continue
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("Telnyx STT message: %s", parsed)
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
            logger.exception("Telnyx STT receive loop error")

        # The server closes the socket right after an error payload, so the
        # close is that error's consequence rather than a separate failure.
        if not self.closed and not self._error_reported:
            self._emit_error_event(
                ConnectionError("Telnyx STT WebSocket closed unexpectedly"),
                context="telnyx_ws_closed",
            )

    def _handle_message(self, data: dict[str, object]) -> None:
        """Dispatch a parsed Telnyx WebSocket message.

        Transcripts arrive as
        ``{"transcript": str, "confidence": float | None, "is_final": bool}``.
        Parameter rejections arrive as ``{"errors": [{"detail": ...}, ...]}``
        and are followed by the server closing the connection.
        """
        errors = data.get("errors")
        if isinstance(errors, list) and errors:
            details = "; ".join(
                str(err.get("detail") or err.get("title"))
                for err in errors
                if isinstance(err, dict)
            )
            self._error_reported = True
            self._emit_error_event(
                RuntimeError(details or "Telnyx STT error"),
                context="telnyx_streaming",
            )
            return

        text = data.get("transcript")
        if not isinstance(text, str) or not text.strip():
            return

        participant = self._current_participant
        if participant is None:
            logger.warning("Telnyx transcript received but no participant set")
            return

        processing_time_ms: Optional[float] = None
        if self._audio_start_time is not None:
            processing_time_ms = (time.perf_counter() - self._audio_start_time) * 1000

        confidence = data.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            confidence = None
        is_final = bool(data.get("is_final", True))

        response = TranscriptResponse(
            confidence=float(confidence) if confidence is not None else None,
            language=self.language,
            model_name=self.model or self.transcription_engine,
            processing_time_ms=processing_time_ms,
        )

        if is_final:
            self._audio_start_time = None
            self._emit_transcript_event(text, participant, response)
        else:
            self._emit_transcript_event(text, participant, response, mode="replacement")
