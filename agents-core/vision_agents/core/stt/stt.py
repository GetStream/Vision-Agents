import abc
import logging
import time
import uuid
from typing import Optional, Dict, Any, Tuple, List, Union
from getstream.video.rtc.track_util import PcmData

from ..edge.types import Participant
from vision_agents.core.events import (
    PluginInitializedEvent,
    PluginClosedEvent,
)
from vision_agents.core.events.manager import EventManager
from . import events

logger = logging.getLogger(__name__)


class STT(abc.ABC):
    """
    Abstract base class for Speech-to-Text implementations.

    This class provides a standardized interface for STT implementations with consistent
    event emission patterns and error handling.

    Events:
        - transcript: Emitted when a complete transcript is available.
            Args: text (str), user_metadata (dict), metadata (dict)
        - partial_transcript: Emitted when a partial transcript is available.
            Args: text (str), user_metadata (dict), metadata (dict)
        - error: Emitted when an error occurs during transcription.
            Args: error (Exception)

    Standard Error Handling:
        - All implementations should catch exceptions in _process_audio_impl and emit error events
        - Use _emit_error_event() helper for consistent error emission
        - Log errors with appropriate context using the logger

    Standard Event Emission:
        - Use _emit_transcript_event() and _emit_partial_transcript_event() helpers
        - Include processing time and audio duration in metadata when available
        - Maintain consistent metadata structure across implementations
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        *,
        provider_name: Optional[str] = None,
    ):
        """
        Initialize the STT service.

        Args:
            sample_rate: The sample rate of the audio to process, in Hz.
            provider_name: Name of the STT provider (e.g., "deepgram", "moonshine")
       """

        self._track = None
        self.sample_rate = sample_rate
        self._is_closed = False
        self.session_id = str(uuid.uuid4())
        self.provider_name = provider_name or self.__class__.__name__
        self.events = EventManager()
        self.events.register_events_from_module(events, ignore_not_compatible=True)

        self.events.send(PluginInitializedEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            plugin_type="STT",
            provider=self.provider_name,
            configuration={"sample_rate": sample_rate},
        ))

    def _validate_pcm_data(self, pcm_data: PcmData) -> bool:
        """
        Validate PCM data input for processing.

        Args:
            pcm_data: The PCM audio data to validate.

        Returns:
            True if the data is valid, False otherwise.
        """

        if not hasattr(pcm_data, "samples") or pcm_data.samples is None:
            logger.warning("PCM data has no samples")
            return False

        if not hasattr(pcm_data, "sample_rate") or pcm_data.sample_rate <= 0:
            logger.warning("PCM data has invalid sample rate")
            return False

        # Check if samples are empty
        if hasattr(pcm_data.samples, "__len__") and len(pcm_data.samples) == 0:
            logger.debug("Received empty audio samples")
            return False

        return True

    def _emit_transcript_event(
        self,
        text: str,
        user_metadata: Optional[Union[Dict[str, Any], Participant]],
        metadata: Dict[str, Any],
    ):
        """
        Emit a final transcript event with structured data.

        Args:
            text: The transcribed text.
            user_metadata: User-specific metadata.
            metadata: Transcription metadata (processing time, confidence, etc.).
        """
        self.events.send(events.STTTranscriptEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            text=text,
            user_metadata=user_metadata,
            confidence=metadata.get("confidence"),
            language=metadata.get("language"),
            processing_time_ms=metadata.get("processing_time_ms"),
            audio_duration_ms=metadata.get("audio_duration_ms"),
            model_name=metadata.get("model_name"),
            words=metadata.get("words"),
        ))

    def _emit_partial_transcript_event(
        self,
        text: str,
        user_metadata: Optional[Union[Dict[str, Any], Participant]],
        metadata: Dict[str, Any],
    ):
        """
        Emit a partial transcript event with structured data.

        Args:
            text: The partial transcribed text.
            user_metadata: User-specific metadata.
            metadata: Transcription metadata (processing time, confidence, etc.).
        """
        self.events.send(events.STTPartialTranscriptEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            text=text,
            user_metadata=user_metadata,
            confidence=metadata.get("confidence"),
            language=metadata.get("language"),
            processing_time_ms=metadata.get("processing_time_ms"),
            audio_duration_ms=metadata.get("audio_duration_ms"),
            model_name=metadata.get("model_name"),
            words=metadata.get("words"),
        ))

    def _emit_error_event(
        self,
        error: Exception,
        context: str = "",
        user_metadata: Optional[Union[Dict[str, Any], Participant]] = None,
    ):
        """
        Emit an error event with structured data.

        Args:
            error: The exception that occurred.
            context: Additional context about where the error occurred.
            user_metadata: User-specific metadata.
        """
        self.events.send(events.STTErrorEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            error=error,
            context=context,
            user_metadata=user_metadata,
            error_code=getattr(error, "error_code", None),
            is_recoverable=not isinstance(error, (SystemExit, KeyboardInterrupt)),
        ))

    async def process_audio(
        self, pcm_data: PcmData, participant: Optional[Participant] = None
    ):
        """
        Process audio data for transcription and emit appropriate events.

        Args:
            pcm_data: The PCM audio data to process.
            user_metadata: Additional metadata about the user or session.
        """
        if self._is_closed:
            logger.debug("Ignoring audio processing request - STT is closed")
            return

        # Validate input data
        if not self._validate_pcm_data(pcm_data):
            logger.warning("Invalid PCM data received, skipping processing")
            return

        try:
            # Process the audio data using the implementation-specific method
            audio_duration_ms = (
                pcm_data.duration * 1000 if hasattr(pcm_data, "duration") else None
            )
            logger.debug(
                "Processing audio chunk",
                extra={
                    "duration_ms": audio_duration_ms,
                    "has_user_metadata": participant is not None,
                },
            )

            start_time = time.time()
            results = await self._process_audio_impl(pcm_data, participant)
            processing_time = time.time() - start_time

            # If no results were returned, just return
            if not results:
                logger.debug(
                    "No speech detected in audio",
                    extra={
                        "processing_time_ms": processing_time * 1000,
                        "audio_duration_ms": audio_duration_ms,
                    },
                )
                return

            # Process each result and emit the appropriate event
            for is_final, text, metadata in results:
                # Ensure metadata includes processing time if not already present
                if "processing_time_ms" not in metadata:
                    metadata["processing_time_ms"] = processing_time * 1000

                if is_final:
                    self._emit_transcript_event(text, participant, metadata)
                else:
                    self._emit_partial_transcript_event(text, participant, metadata)

        except Exception as e:
            # Emit any errors that occur during processing
            self._emit_error_event(e, "audio processing", participant)

    @abc.abstractmethod
    async def _process_audio_impl(
        self, pcm_data: PcmData, user_metadata: Optional[Union[Dict[str, Any], Participant]] = None
    ) -> Optional[List[Tuple[bool, str, Dict[str, Any]]]]:
        """
        Implementation-specific method to process audio data.

        This method must be implemented by all STT providers and should handle the core
        transcription logic. The base class handles event emission and error handling.

        Args:
            pcm_data: The PCM audio data to process. Guaranteed to be valid by base class.
            user_metadata: Additional metadata about the user or session.

        Returns:
            optional list[tuple[bool, str, dict]] | None
                • synchronous providers: a list of results.
                • asynchronous providers: None (they emit events themselves).

        Notes:
            Implementations must not both emit events and return non-empty results,
            or duplicate events will be produced.
            Exceptions should bubble up; process_audio() will catch them
            and emit a single "error" event.
        """
        pass

    @abc.abstractmethod
    async def close(self):
        """
        Close the STT service and release any resources.

        Implementations should:
        - Set self._is_closed = True
        - Clean up any background tasks or connections
        - Release any allocated resources
        - Log the closure appropriately
        """
        if not self._is_closed:
            self._is_closed = True

            # Emit closure event
            self.events.send(PluginClosedEvent(
                session_id=self.session_id,
                plugin_name=self.provider_name,
                plugin_type="STT",
                provider=self.provider_name,
                cleanup_successful=True,
            ))
