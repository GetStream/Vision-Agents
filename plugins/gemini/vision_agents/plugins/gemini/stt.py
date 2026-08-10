import asyncio
import logging
import os
import time
from typing import AsyncContextManager, Optional

from getstream.video.rtc.track_util import PcmData
from google import genai
from google.genai.live import AsyncSession
from google.genai.types import (
    AudioTranscriptionConfigDict,
    Blob,
    HttpOptions,
    LiveConnectConfigDict,
    LiveServerMessage,
    Modality,
    Transcription,
)
from vision_agents.core import stt
from vision_agents.core.edge.types import Participant
from vision_agents.core.stt import TranscriptResponse
from vision_agents.core.utils.utils import cancel_and_wait

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.5-transcribe-live-preview"
FINAL_TRANSCRIPT_DELAY_SECONDS = 0.8


class STT(stt.STT):
    """Gemini Live streaming speech-to-text implementation."""

    turn_detection: bool = True

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        language_codes: Optional[list[str]] = None,
        custom_vocabulary: Optional[list[str]] = None,
        http_options: Optional[HttpOptions] = None,
        client: Optional[genai.Client] = None,
    ) -> None:
        """Initialize Gemini Live STT.

        Args:
            api_key: Gemini API key. Falls back to ``GOOGLE_API_KEY`` or
                ``GEMINI_API_KEY``.
            model: Gemini Live transcription model.
            language_codes: Optional BCP-47 language hints. An empty list
                enables automatic language detection.
            custom_vocabulary: Optional phrases used to bias recognition.
            http_options: Optional Gemini HTTP configuration.
            client: Optional preconfigured Gemini client.
        """
        super().__init__(provider_name="gemini")

        resolved_api_key = (
            api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        )
        if client is None and not resolved_api_key:
            raise ValueError(
                "GOOGLE_API_KEY or GEMINI_API_KEY env var, api_key, or client required"
            )

        self.model = model
        self.language_codes = language_codes
        self.custom_vocabulary = custom_vocabulary

        if client is not None:
            self._client = client
        else:
            self._client = genai.Client(
                api_key=resolved_api_key,
                http_options=http_options or HttpOptions(api_version="v1alpha"),
            )

        transcription_config = AudioTranscriptionConfigDict(
            language_codes=language_codes or [],
        )
        if custom_vocabulary:
            transcription_config["custom_vocabulary"] = custom_vocabulary
        self._config = LiveConnectConfigDict(
            response_modalities=[Modality.TEXT],
            input_audio_transcription=transcription_config,
        )

        self._session: AsyncSession | None = None
        self._session_context: AsyncContextManager[AsyncSession] | None = None
        self._listen_task: asyncio.Task[None] | None = None
        self._finalize_task: asyncio.Task[None] | None = None
        self._current_participant: Participant | None = None
        self._audio_start_time: float | None = None
        self._audio_duration_ms = 0.0
        self._transcript_parts: list[str] = []
        self._interim_text = ""
        self._last_transcription: Transcription | None = None
        self._turn_in_progress = False
        self._received_interim = False
        self._connected = False

    async def start(self) -> None:
        """Open the Gemini Live transcription session."""
        if self.closed:
            raise ValueError("STT is closed and cannot be started")

        await super().start()
        try:
            self._session_context = self._client.aio.live.connect(
                model=self.model,
                config=self._config,
            )
            self._session = await self._session_context.__aenter__()
            self._listen_task = asyncio.create_task(self._listen())
            self._connected = True
            self._on_connected()
        except Exception as exc:
            self.started = False
            self._emit_error_event(exc, context="connect")
            await self._close_session()
            raise

    async def clear(self) -> None:
        """Clear pending transcript and turn state."""
        await self._cancel_finalize_task()
        await super().clear()
        self._reset_turn()
        self._current_participant = None

    async def close(self) -> None:
        """Close the Gemini Live session and background task."""
        if self.closed:
            return

        if self._listen_task is not None:
            await cancel_and_wait(self._listen_task)
            self._listen_task = None

        await self._cancel_finalize_task()
        await self._close_session()
        self._set_disconnected(clean=True)
        self._reset_turn()
        self._current_participant = None
        await super().close()

    async def process_audio(
        self,
        pcm_data: PcmData,
        participant: Optional[Participant] = None,
    ) -> None:
        """Stream PCM audio to Gemini for transcription."""
        if self.closed:
            logger.warning("Gemini STT is closed, ignoring audio")
            return
        if not self.started or self._session is None:
            raise RuntimeError("Gemini STT is not started; call start() first")

        resampled = pcm_data.resample(16_000, 1)
        self._current_participant = participant
        if self._audio_start_time is None:
            self._audio_start_time = time.perf_counter()
        self._audio_duration_ms += resampled.duration * 1000

        await self._session.send_realtime_input(
            audio=Blob(
                data=resampled.samples.tobytes(),
                mime_type="audio/pcm;rate=16000",
            )
        )

    async def _listen(self) -> None:
        session = self._session
        if session is None:
            return

        try:
            async for message in session.receive():
                self._handle_message(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self.closed:
                logger.exception("Gemini STT receive loop failed")
                self.started = False
                self._emit_error_event(exc, context="receive")
                self._set_disconnected(reason=str(exc), clean=False)
        else:
            if not self.closed:
                self.started = False
                self._set_disconnected(clean=True)

    def _handle_message(self, message: LiveServerMessage) -> None:
        server_content = message.server_content
        if server_content is None:
            return

        interim = server_content.interim_input_transcription
        if interim is not None:
            self._handle_transcription(interim, interim=True)

        transcription = server_content.input_transcription
        if transcription is not None:
            self._handle_transcription(transcription, interim=False)

        if server_content.turn_complete:
            self._finish_turn(self._last_transcription)

    def _handle_transcription(
        self,
        transcription: Transcription,
        *,
        interim: bool,
    ) -> None:
        text = transcription.text or ""
        if not text.strip():
            if transcription.finished:
                self._finish_turn(transcription)
            return

        participant = self._current_participant
        if participant is None:
            logger.warning("Received Gemini transcript but no participant set")
            return

        if not self._turn_in_progress:
            self._turn_in_progress = True
            self._emit_turn_started_event(participant)

        self._last_transcription = transcription
        if interim:
            self._received_interim = True
            self._interim_text = text
            self._emit_transcript_event(
                text,
                participant,
                self._build_response(transcription),
                mode="replacement",
            )
            if transcription.finished:
                self._finish_turn(transcription)
            else:
                self._schedule_turn_finalization()
            return

        self._transcript_parts.append(text)
        if transcription.finished:
            self._finish_turn(transcription)
        else:
            if not self._received_interim:
                self._emit_transcript_event(
                    text,
                    participant,
                    self._build_response(transcription),
                    mode="delta",
                )
            self._schedule_turn_finalization()

    def _finish_turn(self, transcription: Transcription | None) -> None:
        if not self._turn_in_progress:
            return

        participant = self._current_participant
        if participant is None:
            return

        text = "".join(self._transcript_parts).strip() or self._interim_text.strip()
        if text:
            response = self._build_response(transcription)
            self._emit_transcript_event(text, participant, response, mode="final")
            self._emit_turn_ended_event(
                participant,
                duration_ms=response.audio_duration_ms,
            )
        self._cancel_finalize_task_nowait()
        self._reset_turn()

    def _build_response(
        self,
        transcription: Transcription | None,
    ) -> TranscriptResponse:
        processing_time_ms: float | None = None
        if self._audio_start_time is not None:
            processing_time_ms = (time.perf_counter() - self._audio_start_time) * 1000

        language = None
        other: dict[str, object] = {}
        if transcription is not None:
            language = transcription.language_code
            if transcription.speaker_label:
                other["speaker_label"] = transcription.speaker_label
            if transcription.words:
                other["words"] = [
                    word.model_dump(mode="json", exclude_none=True)
                    for word in transcription.words
                ]
        if language is None and self.language_codes and len(self.language_codes) == 1:
            language = self.language_codes[0]

        return TranscriptResponse(
            language=language,
            processing_time_ms=processing_time_ms,
            audio_duration_ms=self._audio_duration_ms or None,
            model_name=self.model,
            other=other or None,
        )

    def _reset_turn(self) -> None:
        self._audio_start_time = None
        self._audio_duration_ms = 0.0
        self._transcript_parts.clear()
        self._interim_text = ""
        self._last_transcription = None
        self._turn_in_progress = False
        self._received_interim = False

    def _schedule_turn_finalization(self) -> None:
        self._cancel_finalize_task_nowait()
        self._finalize_task = asyncio.create_task(self._finalize_after_delay())

    async def _finalize_after_delay(self) -> None:
        await asyncio.sleep(FINAL_TRANSCRIPT_DELAY_SECONDS)
        self._finish_turn(self._last_transcription)

    async def _cancel_finalize_task(self) -> None:
        task = self._finalize_task
        self._finalize_task = None
        if task is not None and task is not asyncio.current_task():
            await cancel_and_wait(task)

    def _cancel_finalize_task_nowait(self) -> None:
        task = self._finalize_task
        self._finalize_task = None
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()

    async def _close_session(self) -> None:
        if self._session_context is not None:
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception:
                logger.exception("Failed to close Gemini STT session")
            finally:
                self._session_context = None
                self._session = None

    def _set_disconnected(
        self,
        reason: Optional[str] = None,
        clean: bool = True,
    ) -> None:
        if self._connected:
            self._connected = False
            self._on_disconnected(reason=reason, clean=clean)
