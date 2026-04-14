"""Sarvam AI Speech-to-Text via WebSocket streaming.

Docs: https://docs.sarvam.ai/api-reference-docs/api-guides-tutorials/speech-to-text/streaming-api

Supported models:
    - ``saaras:v3`` (default, recommended) – transcription + translation
    - ``saarika:v2.5`` – legacy transcription-only
    - ``saaras:v2.5`` – legacy translation
"""

import asyncio
import base64
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

WS_BASE_URL = "wss://api.sarvam.ai/speech-to-text/ws"

SUPPORTED_SAMPLE_RATES = {8000, 16000}
SUPPORTED_MODELS = {"saaras:v3", "saarika:v2.5", "saaras:v2.5"}
SUPPORTED_MODES = {"transcribe", "translate", "verbatim", "translit", "codemix"}


class STT(stt.STT):
    """Sarvam AI streaming Speech-to-Text.

    Uses aiohttp for a fully-async WebSocket connection to Sarvam's streaming
    endpoint. Audio is sent as base64-encoded PCM inside JSON messages.
    Transcript and VAD events are emitted as STT and turn events.

    Turn detection is supported natively via Sarvam's VAD signals
    (``speech_start`` / ``speech_end``).
    """

    turn_detection: bool = True

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "saaras:v3",
        language: Optional[str] = None,
        mode: Optional[str] = None,
        sample_rate: int = 16000,
        high_vad_sensitivity: bool = False,
        vad_signals: bool = True,
        prompt: Optional[str] = None,
    ) -> None:
        """Initialize Sarvam STT.

        Args:
            api_key: Sarvam API key. Falls back to ``SARVAM_API_KEY`` env var.
            model: Streaming model id. Defaults to ``saaras:v3``.
            language: Language code (e.g. ``hi-IN``, ``en-IN``). ``None`` lets
                Sarvam auto-detect.
            mode: One of ``transcribe``, ``translate``, ``verbatim``,
                ``translit``, ``codemix``. Saaras defaults are model-dependent.
            sample_rate: Input sample rate, 8000 or 16000 Hz.
            high_vad_sensitivity: Increase VAD sensitivity for noisy input.
            vad_signals: Emit ``speech_start`` / ``speech_end`` events used
                for turn detection.
            prompt: Optional biasing prompt sent once after connect.
        """
        super().__init__(provider_name="sarvam")

        if model not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported Sarvam STT model '{model}'. "
                f"Expected one of: {sorted(SUPPORTED_MODELS)}"
            )
        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"Unsupported sample_rate {sample_rate}. "
                f"Expected one of: {sorted(SUPPORTED_SAMPLE_RATES)}"
            )
        if mode is not None and mode not in SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported mode '{mode}'. Expected one of: {sorted(SUPPORTED_MODES)}"
            )

        self._api_key = api_key or os.environ.get("SARVAM_API_KEY")
        if not self._api_key:
            raise ValueError(
                "SARVAM_API_KEY env var or api_key parameter required for Sarvam STT"
            )

        self.model = model
        self.language = language
        self.mode = mode
        self.sample_rate = sample_rate
        self.high_vad_sensitivity = high_vad_sensitivity
        self.vad_signals = vad_signals
        self._prompt = prompt

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._receive_task: Optional[asyncio.Task[Any]] = None
        self._connection_ready = asyncio.Event()
        self._current_participant: Optional[Participant] = None
        self._audio_start_time: Optional[float] = None

    def _build_ws_url(self) -> str:
        params: dict[str, str | int] = {
            "model": self.model,
            "sample_rate": self.sample_rate,
            "vad_signals": "true" if self.vad_signals else "false",
        }
        if self.language is not None:
            params["language-code"] = self.language
        if self.mode is not None:
            params["mode"] = self.mode
        if self.high_vad_sensitivity:
            params["high_vad_sensitivity"] = "true"
        return f"{WS_BASE_URL}?{urlencode(params)}"

    async def start(self) -> None:
        """Open the Sarvam WebSocket and start the receive loop."""
        await super().start()

        url = self._build_ws_url()
        headers = {"api-subscription-key": self._api_key or ""}

        self._session = aiohttp.ClientSession()
        self._ws = await self._session.ws_connect(url, headers=headers)

        if self._prompt:
            await self._ws.send_str(json.dumps({"config": {"prompt": self._prompt}}))

        self._receive_task = asyncio.create_task(self._receive_loop())
        self._connection_ready.set()

    async def process_audio(
        self,
        pcm_data: PcmData,
        participant: Optional[Participant] = None,
    ) -> None:
        """Send a PCM audio chunk to Sarvam.

        The chunk is resampled to the configured sample rate and wrapped in
        the JSON schema expected by Sarvam's WebSocket.
        """
        if self.closed:
            logger.warning("Sarvam STT is closed, ignoring audio")
            return

        await self._connection_ready.wait()

        if self._ws is None or self._ws.closed:
            logger.warning("Sarvam STT WebSocket not open, dropping audio")
            return

        resampled = pcm_data.resample(self.sample_rate, 1)
        audio_bytes = resampled.samples.tobytes()

        self._current_participant = participant
        if self._audio_start_time is None:
            self._audio_start_time = time.perf_counter()

        message = {
            "audio": {
                "data": base64.b64encode(audio_bytes).decode("ascii"),
                "encoding": "audio/wav",
                "sample_rate": self.sample_rate,
            }
        }
        await self._ws.send_str(json.dumps(message))

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
                        logger.warning("Sarvam STT sent non-JSON text: %s", msg.data)
                        continue
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("Sarvam STT message: %s", parsed)
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
            logger.exception("Sarvam STT receive loop error")

        if not self.closed:
            self._emit_error_event(
                ConnectionError("Sarvam STT WebSocket closed unexpectedly"),
                self._current_participant,
                "sarvam_ws_closed",
            )

    def _handle_message(self, data: dict[str, Any]) -> None:
        """Dispatch a parsed Sarvam WebSocket message.

        Sarvam's streaming STT sends three message shapes:

        - ``{"type": "events", "data": {"signal_type": "START_SPEECH" | "END_SPEECH"}}``
          VAD boundaries used to drive turn events.
        - ``{"type": "data", "data": {"transcript": "...", "language_code": ...}}``
          The finalized transcript for the current utterance (no partials).
        - ``{"type": "error", ...}`` or any message with an ``error`` key.
        """
        msg_type = data.get("type", "")
        payload = data.get("data") or {}
        participant = self._current_participant

        if msg_type == "events":
            signal = payload.get("signal_type", "")
            if participant is None:
                return
            if signal == "START_SPEECH":
                self._emit_turn_started_event(participant)
            elif signal == "END_SPEECH":
                self._audio_start_time = None
                self._emit_turn_ended_event(participant)
            return

        if msg_type == "error" or "error" in data:
            err_msg = data.get("error") or payload.get("message") or "Sarvam STT error"
            self._emit_error_event(
                Exception(str(err_msg)),
                participant,
                "sarvam_streaming",
            )
            return

        transcript_text = payload.get("transcript") or data.get("transcript") or ""
        if not transcript_text:
            return

        if participant is None:
            logger.warning("Sarvam transcript received but no participant set")
            return

        processing_time_ms: Optional[float] = None
        if self._audio_start_time is not None:
            processing_time_ms = (time.perf_counter() - self._audio_start_time) * 1000

        language_code = (
            payload.get("language_code")
            or data.get("language_code")
            or self.language
            or "auto"
        )
        metrics = payload.get("metrics") or {}
        audio_duration = metrics.get("audio_duration")
        audio_duration_ms: Optional[int] = (
            int(audio_duration * 1000) if audio_duration is not None else None
        )

        response = TranscriptResponse(
            language=language_code,
            model_name=self.model,
            processing_time_ms=processing_time_ms,
            audio_duration_ms=audio_duration_ms,
        )

        # Sarvam streaming only sends finalized transcripts (per utterance),
        # so treat every `type: data` message as a final transcript event.
        self._emit_transcript_event(transcript_text, participant, response)

    async def close(self) -> None:
        """Send end_of_stream, close the WebSocket, and clean up."""
        await super().close()

        if self._ws is not None and not self._ws.closed:
            try:
                await self._ws.send_str(json.dumps({"type": "end_of_stream"}))
            except (aiohttp.ClientError, ConnectionError):
                logger.debug("Could not send end_of_stream to Sarvam")
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
        self._audio_start_time = None
