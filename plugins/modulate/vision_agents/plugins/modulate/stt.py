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
from vision_agents.core.utils.utils import cancel_and_wait

logger = logging.getLogger(__name__)

WS_URL = "wss://platform.modulate.ai/api/velma-2-stt-streaming"
MODEL_NAME = "velma-2"


class STT(stt.STT):
    """Modulate AI Velma-2 streaming Speech-to-Text.

    Uses aiohttp for a fully async WebSocket connection to Modulate AI's
    streaming endpoint with built-in speaker diarization and turn detection.

    Docs: https://docs.modulate.ai/api-reference/stt/streaming
    """

    turn_detection: bool = True

    def __init__(
        self,
        api_key: Optional[str] = None,
        sample_rate: int = 16000,
        speaker_diarization: bool = True,
        emotion_signal: bool = False,
        accent_signal: bool = False,
        deepfake_signal: bool = False,
        pii_phi_tagging: bool = False,
        partial_results: bool = False,
        **kwargs: str | int | bool,
    ):
        """Initialize Modulate AI STT.

        Args:
            api_key: Modulate AI API key. Falls back to MODULATE_API_KEY env var.
            sample_rate: Audio sample rate in Hz. Defaults to 16000.
            speaker_diarization: Label distinct speakers. Defaults to True.
            emotion_signal: Include per-utterance emotion classification.
            accent_signal: Include per-utterance accent classification.
            deepfake_signal: Include synthetic-voice deepfake scoring.
            pii_phi_tagging: Wrap sensitive data in PII/PHI tags.
            partial_results: Stream interim partial transcripts before each final utterance.
            **kwargs: Additional query parameters forwarded to the streaming endpoint.
        """
        super().__init__(provider_name="modulate")

        self._api_key = api_key or os.environ.get("MODULATE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "api_key is required. Pass it directly or set MODULATE_API_KEY."
            )

        self._sample_rate = sample_rate
        self._speaker_diarization = speaker_diarization
        self._emotion_signal = emotion_signal
        self._accent_signal = accent_signal
        self._deepfake_signal = deepfake_signal
        self._pii_phi_tagging = pii_phi_tagging
        self._partial_results = partial_results
        self._extra_params = kwargs

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._connection_ready = asyncio.Event()
        self._done_received = asyncio.Event()
        self._current_participant: Optional[Participant] = None
        self._audio_start_time: Optional[float] = None
        self._speaker_participants: dict[int, Participant] = {}

    def _build_ws_url(self) -> str:
        # Modulate's streaming API authenticates via an api_key query parameter
        # (no Bearer-header alternative), so the key may appear in server/proxy
        # logs. This is required by the endpoint, not the same as header auth.
        params: dict[str, str | int] = {
            "api_key": self._api_key or "",
            "audio_format": "raw",
            "sample_rate": self._sample_rate,
            "num_channels": 1,
            "speaker_diarization": str(self._speaker_diarization).lower(),
            "emotion_signal": str(self._emotion_signal).lower(),
            "accent_signal": str(self._accent_signal).lower(),
            "deepfake_signal": str(self._deepfake_signal).lower(),
            "pii_phi_tagging": str(self._pii_phi_tagging).lower(),
            "partial_results": str(self._partial_results).lower(),
        }
        for key, value in self._extra_params.items():
            params[key] = str(value).lower() if isinstance(value, bool) else value
        return f"{WS_URL}?{urlencode(params)}"

    async def start(self):
        """Start the Modulate AI WebSocket connection and begin listening."""
        await super().start()

        self._session = aiohttp.ClientSession()
        try:
            url = self._build_ws_url()
            self._ws = await self._session.ws_connect(url)
        except Exception:
            await self._session.close()
            self._session = None
            raise
        self._receive_task = asyncio.create_task(self._receive_loop())
        self._connection_ready.set()
        self._on_connected()
        logger.info("Modulate AI WebSocket connection established")

    async def _receive_loop(self) -> None:
        """Read and dispatch incoming WebSocket messages."""
        if self._ws is None:
            return
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    self._handle_message(json.loads(msg.data))
                elif msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                    aiohttp.WSMsgType.ERROR,
                ):
                    break
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Error in Modulate AI receive loop")

        if not self.closed:
            logger.warning("Modulate AI WebSocket closed unexpectedly")
            self._on_disconnected(clean=False)

    def _handle_message(self, data: dict[str, object]) -> None:
        """Dispatch a parsed JSON message by its type field."""
        msg_type = data.get("type", "")

        if msg_type == "utterance":
            utterance = data.get("utterance", {})
            if isinstance(utterance, dict):
                self._handle_utterance(utterance)
        elif msg_type == "partial_utterance":
            utterance = data.get("utterance", {})
            if isinstance(utterance, dict):
                self._handle_partial_utterance(utterance)
        elif msg_type == "done":
            logger.info(
                "Modulate AI session done: %sms audio processed",
                data.get("duration_ms", 0),
            )
            self._done_received.set()
        elif msg_type == "error":
            logger.error("Modulate AI streaming error: %s", data.get("error"))
        else:
            logger.debug("Unhandled Modulate AI message type: %s", msg_type)

    def _resolve_participant(self, speaker: Optional[int]) -> Optional[Participant]:
        """Map a speaker index to a Participant, creating a synthetic one if needed."""
        if not self._speaker_diarization or speaker is None:
            return self._current_participant

        cached = self._speaker_participants.get(speaker)
        if cached is not None:
            return cached

        participant = Participant(
            original=None,
            user_id=f"speaker_{speaker}",
            id=f"speaker_{speaker}_{self.session_id[:8]}",
        )
        self._speaker_participants[speaker] = participant
        return participant

    def _handle_utterance(self, utterance: dict[str, object]) -> None:
        """Handle a final utterance — emit final transcript and turn ended."""
        text = utterance.get("text")
        if not isinstance(text, str) or not text.strip():
            return
        text = text.strip()

        speaker = utterance.get("speaker")
        participant = self._resolve_participant(
            speaker if isinstance(speaker, int) else None
        )
        if participant is None:
            logger.warning("Received utterance but no participant available")
            return

        processing_time_ms: Optional[float] = None
        if self._audio_start_time is not None:
            processing_time_ms = (time.perf_counter() - self._audio_start_time) * 1000

        other: dict[str, object] = {"utterance_uuid": utterance.get("utterance_uuid")}
        if self._speaker_diarization:
            other["speaker"] = utterance.get("speaker")
        if self._emotion_signal:
            other["emotion"] = utterance.get("emotion")
        if self._accent_signal:
            other["accent"] = utterance.get("accent")
        if self._deepfake_signal:
            other["deepfake_score"] = utterance.get("deepfake_score")

        duration = utterance.get("duration_ms")
        duration_ms = float(duration) if isinstance(duration, (int, float)) else None
        language = utterance.get("language")
        response = TranscriptResponse(
            language=language if isinstance(language, str) else None,
            audio_duration_ms=duration_ms,
            model_name=MODEL_NAME,
            processing_time_ms=processing_time_ms,
            other=other,
        )

        self._emit_transcript_event(text, participant, response, mode="final")
        self._audio_start_time = None
        self._emit_turn_ended_event(participant, duration_ms=duration_ms)

    def _handle_partial_utterance(self, utterance: dict[str, object]) -> None:
        """Handle a partial utterance — emit replacement transcript."""
        text = utterance.get("text")
        if not isinstance(text, str) or not text.strip():
            return
        text = text.strip()

        speaker = utterance.get("speaker")
        participant = self._resolve_participant(
            speaker if isinstance(speaker, int) else None
        )
        if participant is None:
            return

        processing_time_ms: Optional[float] = None
        if self._audio_start_time is not None:
            processing_time_ms = (time.perf_counter() - self._audio_start_time) * 1000

        response = TranscriptResponse(
            model_name=MODEL_NAME,
            processing_time_ms=processing_time_ms,
        )
        self._emit_transcript_event(text, participant, response, mode="replacement")

    async def process_audio(
        self,
        pcm_data: PcmData,
        participant: Participant,
    ):
        """Process audio data through Modulate AI for transcription.

        Args:
            pcm_data: The PCM audio data to process.
            participant: The participant the audio belongs to.
        """
        if self.closed:
            logger.warning("Modulate AI STT is closed, ignoring audio")
            return

        await self._connection_ready.wait()

        if self._ws is None or self._ws.closed:
            logger.warning("Modulate AI WebSocket not available")
            return

        resampled = pcm_data.resample(self._sample_rate, 1)
        audio_bytes = resampled.samples.tobytes()

        self._current_participant = participant

        if self._audio_start_time is None:
            self._audio_start_time = time.perf_counter()

        await self._ws.send_bytes(audio_bytes)

    async def close(self):
        """Close the Modulate AI WebSocket connection and clean up resources."""
        await super().close()

        if self._ws is not None and not self._ws.closed:
            try:
                # Signal end of audio stream
                await self._ws.send_str("")
            except Exception:
                logger.debug("Could not send end-of-stream signal")

            try:
                await asyncio.wait_for(self._done_received.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.debug("Timeout waiting for Modulate AI done event")

        if self._receive_task is not None:
            await cancel_and_wait(self._receive_task)
            self._receive_task = None

        if self._ws is not None and not self._ws.closed:
            try:
                await self._ws.close()
            except Exception:
                logger.debug("Error closing Modulate AI WebSocket")
        self._ws = None

        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

        self._connection_ready.clear()
        self._done_received.clear()
        self._audio_start_time = None
        self._speaker_participants.clear()
        self._on_disconnected()
