"""Sarvam AI Speech-to-Text via WebSocket streaming.

Docs:
    - Realtime: https://docs.sarvam.ai/api/api-guides-tutorials/speech-to-text/realtime-streaming
    - Legacy: https://docs.sarvam.ai/api/api-guides-tutorials/speech-to-text/streaming-api

Supported models:
    - ``saaras:v3-realtime`` (default) – low-latency realtime streaming
    - ``saaras:v3`` – legacy WebSocket transcription + translation
    - ``saaras:v4`` – latest transcription model on the legacy WebSocket
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

WS_STT_URL = "wss://api.sarvam.ai/speech-to-text/ws"
WS_STT_REALTIME_URL = "wss://api.sarvam.ai/speech-to-text-realtime/ws"
REALTIME_MODEL = "saaras:v3-realtime"
KEEPALIVE_INTERVAL_S = 20

SUPPORTED_SAMPLE_RATES = {8000, 16000}
SUPPORTED_MODES = {"transcribe", "translate", "verbatim", "translit", "codemix"}
SUPPORTED_MODELS = {"saaras:v3", REALTIME_MODEL, "saaras:v4"}
SUPPORTED_STREAM_TYPES = {"fast", "balanced", "simulated"}


class STT(stt.STT):
    """Sarvam AI streaming Speech-to-Text.

    Uses aiohttp for a fully-async WebSocket connection. ``saaras:v3-realtime``
    uses ``/speech-to-text-realtime/ws``; other models use the legacy
    ``/speech-to-text/ws`` endpoint.
    """

    turn_detection: bool = True

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = REALTIME_MODEL,
        language: Optional[str] = None,
        mode: Optional[str] = None,
        sample_rate: int = 16000,
        high_vad_sensitivity: bool = False,
        vad_signals: bool = True,
        prompt: Optional[str] = None,
        stream_type: str = "fast",
    ) -> None:
        """Initialize Sarvam STT.

        Args:
            api_key: Sarvam API key. Falls back to ``SARVAM_API_KEY`` env var.
            model: Streaming model id. Defaults to ``saaras:v3-realtime``.
            language: Language code (e.g. ``hi-IN``, ``en-IN``). ``None`` lets
                Sarvam auto-detect.
            mode: One of ``transcribe``, ``translate``, ``verbatim``,
                ``translit``, ``codemix``. Saaras defaults are model-dependent.
            sample_rate: Input sample rate, 8000 or 16000 Hz.
            high_vad_sensitivity: Increase VAD sensitivity for noisy input.
                Only used on the legacy WebSocket.
            vad_signals: Emit ``speech_start`` / ``speech_end`` events used
                for turn detection.
            prompt: Optional biasing prompt.
            stream_type: Realtime latency/accuracy tradeoff (``fast``,
                ``balanced``, ``simulated``). Only used for
                ``saaras:v3-realtime``.
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
        if stream_type not in SUPPORTED_STREAM_TYPES:
            raise ValueError(
                f"Unsupported stream_type '{stream_type}'. "
                f"Expected one of: {sorted(SUPPORTED_STREAM_TYPES)}"
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
        self.stream_type = stream_type
        self._prompt = prompt

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._receive_task: Optional[asyncio.Task[None]] = None
        self._keepalive_task: Optional[asyncio.Task[None]] = None
        self._connection_ready = asyncio.Event()
        self._current_participant: Optional[Participant] = None
        self._audio_start_time: Optional[float] = None

        self._in_speech: bool = False
        self._pending_transcript: Optional[str] = None
        self._pending_response: Optional[TranscriptResponse] = None
        self._turn_end_pending: bool = False

    @property
    def _is_realtime(self) -> bool:
        return self.model == REALTIME_MODEL

    def _build_ws_url(self) -> str:
        if self._is_realtime:
            params: dict[str, str | int] = {
                "model": self.model,
                "language_code": self.language or "auto",
                "sample_rate": self.sample_rate,
                "encoding": "linear16",
                "stream_type": self.stream_type,
                "endpointing": "vad",
            }
            if self.mode is not None:
                params["mode"] = self.mode
            if self._prompt:
                params["prompt"] = self._prompt
            return f"{WS_STT_REALTIME_URL}?{urlencode(params)}"

        params = {
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
        return f"{WS_STT_URL}?{urlencode(params)}"

    async def start(self) -> None:
        """Open the Sarvam WebSocket and start the receive loop."""
        await super().start()

        url = self._build_ws_url()
        headers = {"api-subscription-key": self._api_key or ""}

        self._session = aiohttp.ClientSession()
        self._ws = await self._session.ws_connect(url, headers=headers)

        if self._prompt and not self._is_realtime:
            await self._ws.send_str(
                json.dumps({"type": "config", "prompt": self._prompt})
            )

        self._receive_task = asyncio.create_task(self._receive_loop())
        if self._is_realtime:
            self._start_keepalive()
        self._connection_ready.set()
        self._on_connected()

    async def process_audio(
        self,
        pcm_data: PcmData,
        participant: Participant,
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
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

        self._current_participant = participant
        if self._audio_start_time is None:
            self._audio_start_time = time.perf_counter()

        if self._is_realtime:
            await self._ws.send_str(
                json.dumps({"event": "audio_input", "audio": audio_b64})
            )
            return

        await self._ws.send_str(
            json.dumps(
                {
                    "audio": {
                        "data": audio_b64,
                        "encoding": "audio/wav",
                        "sample_rate": self.sample_rate,
                    }
                }
            )
        )

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
                context="sarvam_ws_closed",
            )

    def _handle_message(self, data: dict[str, Any]) -> None:
        """Dispatch a parsed Sarvam WebSocket message."""
        if "event" in data:
            self._handle_realtime_message(data)
            return
        self._handle_legacy_message(data)

    def _handle_realtime_message(self, data: dict[str, Any]) -> None:
        """Handle saaras:v3-realtime events.

        - ``vad.speech_start`` / ``vad.speech_end`` drive turn events.
        - ``transcript.partial`` is a replacement transcript during speech.
        - ``transcript.final`` is the completed utterance.
        - ``error`` carries ``code``, ``is_fatal``, and ``message``.
        """
        event = data.get("event", "")
        participant = self._current_participant

        if event == "error":
            err_msg = data.get("message") or "Sarvam STT error"
            self._emit_error_event(
                Exception(str(err_msg)),
                context="sarvam_streaming",
            )
            return

        if event == "vad.speech_start":
            self._in_speech = True
            self._turn_end_pending = False
            if participant is not None and self.vad_signals:
                self._emit_turn_started_event(participant)
            return

        if event == "vad.speech_end":
            self._in_speech = False
            # The realtime endpoint sends vad.speech_end before
            # transcript.final, so the turn end waits for that transcript.
            self._turn_end_pending = True
            return

        if event not in ("transcript.partial", "transcript.final"):
            return

        transcript_text = data.get("text") or ""
        if not transcript_text:
            return
        if participant is None:
            logger.warning("Sarvam transcript received but no participant set")
            return

        processing_time_ms: Optional[float] = None
        if self._audio_start_time is not None:
            processing_time_ms = (time.perf_counter() - self._audio_start_time) * 1000

        audio_duration_ms: Optional[int] = None
        start_s = data.get("start_s")
        end_s = data.get("end_s")
        if start_s is not None and end_s is not None:
            audio_duration_ms = int((float(end_s) - float(start_s)) * 1000)

        response = TranscriptResponse(
            language=data.get("language") or self.language or "auto",
            model_name=self.model,
            processing_time_ms=processing_time_ms,
            audio_duration_ms=audio_duration_ms,
        )
        if event == "transcript.partial":
            self._emit_transcript_event(
                transcript_text, participant, response, mode="replacement"
            )
            return

        self._emit_transcript_event(transcript_text, participant, response)
        self._audio_start_time = None
        if self._turn_end_pending:
            self._turn_end_pending = False
            if self.vad_signals:
                self._emit_turn_ended_event(participant)

    def _handle_legacy_message(self, data: dict[str, Any]) -> None:
        """Dispatch a parsed legacy Sarvam WebSocket message.

        - ``{"type": "events", "data": {"signal_type": "START_SPEECH" | "END_SPEECH"}}``
          VAD boundaries used to drive turn events.
        - ``{"type": "data", "data": {"transcript": "...", "language_code": ...}}``
          Transcript updates during an utterance. Sarvam may send multiple
          ``data`` messages per utterance as it refines the text. Only the
          last one before ``END_SPEECH`` is treated as final.
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
                self._in_speech = True
                self._pending_transcript = None
                self._pending_response = None
                self._turn_end_pending = False
                self._emit_turn_started_event(participant)
            elif signal == "END_SPEECH":
                self._in_speech = False
                self._audio_start_time = None
                if self._pending_transcript and self._pending_response:
                    self._emit_transcript_event(
                        self._pending_transcript,
                        participant,
                        self._pending_response,
                    )
                    self._pending_transcript = None
                    self._pending_response = None
                    self._emit_turn_ended_event(participant)
                else:
                    self._turn_end_pending = True
            return

        if msg_type == "error" or "error" in data:
            err_msg = data.get("error") or payload.get("message") or "Sarvam STT error"
            self._emit_error_event(
                Exception(str(err_msg)),
                context="sarvam_streaming",
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

        if self._in_speech:
            self._pending_transcript = transcript_text
            self._pending_response = response
            self._emit_transcript_event(
                transcript_text, participant, response, mode="replacement"
            )
        elif self._turn_end_pending:
            self._turn_end_pending = False
            self._emit_transcript_event(transcript_text, participant, response)
            self._emit_turn_ended_event(participant)
        else:
            self._emit_transcript_event(transcript_text, participant, response)

    def _start_keepalive(self) -> None:
        self._stop_keepalive()
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    def _stop_keepalive(self) -> None:
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            self._keepalive_task = None

    async def _keepalive_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(KEEPALIVE_INTERVAL_S)
                if self._ws is None or self._ws.closed:
                    return
                await self._ws.send_str(json.dumps({"event": "ping"}))
        except asyncio.CancelledError:
            pass
        except (aiohttp.ClientError, ConnectionError):
            logger.debug("Sarvam STT keepalive send failed")

    async def close(self) -> None:
        """Send end-of-stream, close the WebSocket, and clean up."""
        await super().close()
        self._stop_keepalive()

        if self._ws is not None and not self._ws.closed:
            end_message = (
                {"event": "end"} if self._is_realtime else {"type": "end_of_stream"}
            )
            try:
                await self._ws.send_str(json.dumps(end_message))
            except (aiohttp.ClientError, ConnectionError):
                logger.debug("Could not send end message to Sarvam")
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
