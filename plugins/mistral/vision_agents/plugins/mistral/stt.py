import asyncio
import logging
import os
import time
from typing import Any, Optional

from getstream.video.rtc.track_util import PcmData
from mistralai import Mistral
from mistralai.extra.realtime import AudioFormat, RealtimeConnection
from mistralai.models import (
    RealtimeTranscriptionError,
    TranscriptionStreamDone,
    TranscriptionStreamTextDelta,
)
from vision_agents.core import stt
from vision_agents.core.edge.types import Participant
from vision_agents.core.stt import TranscriptResponse
from vision_agents.core.utils.utils import cancel_and_wait

logger = logging.getLogger(__name__)


class STT(stt.STT):
    """
    Mistral Voxtral Realtime Speech-to-Text implementation.

    Uses WebSocket streaming for low-latency transcription.

    Docs:
    - https://docs.mistral.ai/capabilities/audio_transcription#realtime-transcription
    """

    turn_detection: bool = False

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "voxtral-mini-transcribe-realtime-2602",
        sample_rate: int = 16000,
        client: Optional[Mistral] = None,
    ):
        """
        Initialize Mistral Voxtral STT.

        Args:
            api_key: Mistral API key. If not provided, MISTRAL_API_KEY env var is used.
            model: Model to use for transcription.
            sample_rate: Audio sample rate in Hz. Supports 8000, 16000, 22050, 44100, 48000.
            client: Optional pre-configured Mistral client instance.
        """
        super().__init__(provider_name="mistral")

        if client is not None:
            self._client = client
        else:
            if not api_key:
                api_key = os.environ.get("MISTRAL_API_KEY")
            self._client = Mistral(api_key=api_key)

        self.model = model
        self.sample_rate = sample_rate
        self._connection: Optional[RealtimeConnection] = None
        self._receive_task: Optional[asyncio.Task[Any]] = None
        self._current_participant: Optional[Participant] = None
        self._connection_ready = asyncio.Event()
        self._audio_start_time: Optional[float] = None
        self._accumulated_text: str = ""
        self._done_received = asyncio.Event()

    async def start(self):
        """Start the Mistral WebSocket connection."""
        await super().start()

        if self._connection is not None:
            logger.warning("Mistral connection already started")
            return

        audio_format = AudioFormat(encoding="pcm_s16le", sample_rate=self.sample_rate)

        self._connection = await asyncio.wait_for(
            self._client.audio.realtime.connect(
                model=self.model,
                audio_format=audio_format,
            ),
            timeout=10.0,
        )

        self._receive_task = asyncio.create_task(self._receive_loop())
        self._connection_ready.set()

        logger.info("Mistral WebSocket connection established")

    async def _receive_loop(self):
        """Background task to receive and process events from Mistral."""
        if self._connection is None:
            return

        try:
            async for event in self._connection:
                logger.debug(f"Mistral event: {type(event).__name__}")

                if isinstance(event, TranscriptionStreamTextDelta):
                    self._handle_text_delta(event)
                elif isinstance(event, TranscriptionStreamDone):
                    self._handle_done(event)
                    break  # Exit loop after done
                elif isinstance(event, RealtimeTranscriptionError):
                    self._handle_error(event)
                    break  # Exit loop on error

        except asyncio.CancelledError:
            logger.debug("Mistral receive loop cancelled")
            raise
        except Exception as e:
            logger.exception("Error in Mistral receive loop")
            if not self.closed:
                self._emit_error_event(
                    e, context="receive_loop", participant=self._current_participant
                )

    def _handle_text_delta(self, event: TranscriptionStreamTextDelta):
        """Handle text delta - emit word-by-word partials, full text on complete."""
        text = event.text
        if not text:
            return

        participant = self._current_participant
        if participant is None:
            logger.warning("Received transcript but no participant set")
            return

        # Accumulate text for complete events
        self._accumulated_text += text

        processing_time_ms: Optional[float] = None
        if self._audio_start_time is not None:
            processing_time_ms = (time.perf_counter() - self._audio_start_time) * 1000

        response = TranscriptResponse(
            model_name=self.model,
            processing_time_ms=processing_time_ms,
        )

        # Emit partial with just the new word/delta (not accumulated)
        text_stripped = text.strip()
        if text_stripped:
            self._emit_partial_transcript_event(text_stripped, participant, response)

        # Check for sentence-ending punctuation - emit complete transcript
        if text.rstrip().endswith((".", "?", "!")):
            accumulated_stripped = self._accumulated_text.strip()
            if accumulated_stripped:
                self._emit_transcript_event(accumulated_stripped, participant, response)
                self._accumulated_text = ""
                self._audio_start_time = None

    def _handle_done(self, event: TranscriptionStreamDone):
        """Handle end-of-stream event with full transcript."""
        text = event.text.strip()
        if not text:
            return

        participant = self._current_participant
        if participant is None:
            logger.warning("Received done event but no participant set")
            return

        response = TranscriptResponse(
            language=event.language,
            model_name=event.model,
        )

        self._emit_transcript_event(text, participant, response)
        self._accumulated_text = ""
        self._audio_start_time = None
        self._done_received.set()

    def _handle_error(self, event: RealtimeTranscriptionError):
        """Handle error event."""
        error_msg = str(event.error) if event.error else "Unknown Mistral error"
        logger.error(f"Mistral transcription error: {error_msg}")

        error = Exception(error_msg)
        self._emit_error_event(
            error, context="transcription", participant=self._current_participant
        )
        self._audio_start_time = None

    async def process_audio(
        self,
        pcm_data: PcmData,
        participant: Participant,
    ):
        """
        Process audio data through Mistral for transcription.

        Args:
            pcm_data: The PCM audio data to process.
            participant: Optional participant metadata.
        """
        if self.closed:
            logger.warning("Mistral STT is closed, ignoring audio")
            return

        await self._connection_ready.wait()

        if self._connection is None or self._connection.is_closed:
            logger.warning("Mistral connection not available")
            return

        resampled = pcm_data.resample(self.sample_rate, 1)
        audio_bytes = resampled.samples.tobytes()

        self._current_participant = participant

        if self._audio_start_time is None:
            self._audio_start_time = time.perf_counter()

        await self._connection.send_audio(audio_bytes)

    async def close(self):
        """Close the Mistral connection and clean up resources."""
        await super().close()

        # Signal end of audio to trigger Done event with full transcript
        if self._connection and not self._connection.is_closed:
            try:
                await self._connection.end_audio()
            except Exception as e:
                logger.warning(f"Error signaling end of audio: {e}")

        # Wait for Done event with timeout
        if self._receive_task and not self._done_received.is_set():
            try:
                await asyncio.wait_for(self._done_received.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.debug("Timeout waiting for done event")

        if self._receive_task:
            await cancel_and_wait(self._receive_task)
            self._receive_task = None

        if self._connection:
            try:
                await self._connection.close()
            except Exception as e:
                logger.warning(f"Error closing Mistral connection: {e}")
            finally:
                self._connection = None
                self._connection_ready.clear()
                self._done_received.clear()

        self._audio_start_time = None
        self._accumulated_text = ""
