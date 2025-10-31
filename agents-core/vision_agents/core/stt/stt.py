import abc
import logging
import uuid
from typing import Optional
from getstream.video.rtc.track_util import PcmData

from ..edge.types import Participant
from vision_agents.core.events.manager import EventManager
from ..observability.metrics import Timer, stt_latency_ms, stt_errors
from . import events
from .events import TranscriptResponse

logger = logging.getLogger(__name__)


class STT(abc.ABC):
    """
    Abstract base class for Speech-to-Text implementations.

    Subclasses implement this and have to call
    - _emit_partial_transcript_event
    - _emit_transcript_event
    - _emit_error_event for temporary errors

    process_audio is currently called every 20ms. The integration with turn keeping could be improved
    """

    closed: bool = False
    started: bool = False

    def __init__(
        self,
        provider_name: Optional[str] = None,
    ):
        self.session_id = str(uuid.uuid4())
        self.provider_name = provider_name or self.__class__.__name__

        self.events = EventManager()
        self.events.register_events_from_module(events, ignore_not_compatible=True)

    def _emit_transcript_event(
        self,
        text: str,
        participant: Participant,
        response: TranscriptResponse,
    ):
        """
        Emit a final transcript event with structured data.

        Args:
            text: The transcribed text.
            participant: Participant metadata.
            response: Transcription response metadata.
        """
        self.events.send(
            events.STTTranscriptEvent(
                session_id=self.session_id,
                plugin_name=self.provider_name,
                text=text,
                participant=participant,
                response=response,
            )
        )

    def _emit_partial_transcript_event(
        self,
        text: str,
        participant: Participant,
        response: TranscriptResponse,
    ):
        """
        Emit a partial transcript event with structured data.

        Args:
            text: The partial transcribed text.
            participant: Participant metadata.
            response: Transcription response metadata.
        """
        self.events.send(
            events.STTPartialTranscriptEvent(
                session_id=self.session_id,
                plugin_name=self.provider_name,
                text=text,
                participant=participant,
                response=response,
            )
        )

    def _emit_error_event(
        self,
        error: Exception,
        context: str = "",
        participant: Optional[Participant] = None,
    ):
        """
        Emit an error event. Note this should only be emitted for temporary errors.
        Permanent errors due to config etc should be directly raised
        """
        # Increment error counter
        stt_errors.add(
            1, {"provider": self.provider_name, "error_type": type(error).__name__}
        )

        self.events.send(
            events.STTErrorEvent(
                session_id=self.session_id,
                plugin_name=self.provider_name,
                error=error,
                context=context,
                participant=participant,
                error_code=getattr(error, "error_code", None),
                is_recoverable=not isinstance(error, (SystemExit, KeyboardInterrupt)),
            )
        )

    async def process_audio(
        self,
        pcm_data: PcmData,
        participant: Optional[Participant] = None,
    ):
        """
        Process audio with automatic metrics tracking.

        This method wraps the actual processing with metrics collection
        and delegates to the _process_audio method that subclasses implement.

        Args:
            pcm_data: Audio data to process
            participant: Optional participant metadata
        """
        with Timer(stt_latency_ms) as timer:
            # Use fully qualified class path for better identification
            timer.attributes["stt_class"] = (
                f"{self.__class__.__module__}.{self.__class__.__qualname__}"
            )
            timer.attributes["provider"] = self.provider_name
            timer.attributes["sample_rate"] = pcm_data.sample_rate
            timer.attributes["channels"] = pcm_data.channels
            timer.attributes["samples"] = (
                len(pcm_data.samples) if pcm_data.samples is not None else 0
            )
            timer.attributes["duration_ms"] = pcm_data.duration_ms

            try:
                await self._process_audio(pcm_data, participant)
            except Exception as e:
                timer.attributes["error"] = type(e).__name__
                raise

    @abc.abstractmethod
    async def _process_audio(
        self,
        pcm_data: PcmData,
        participant: Optional[Participant] = None,
    ):
        """
        Process audio data and emit transcription events.

        Subclasses must implement this method to perform the actual STT processing.
        The base class handles metrics collection automatically.

        Args:
            pcm_data: Audio data to process
            participant: Optional participant metadata
        """
        pass

    async def start(self):
        if self.started:
            raise ValueError("STT is already started, dont call this method twice")
        self.started = True

    async def close(self):
        self.closed = True
