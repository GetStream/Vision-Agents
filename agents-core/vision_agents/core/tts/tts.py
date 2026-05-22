import abc
import logging
import time
import uuid
from dataclasses import dataclass
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Iterator,
    Optional,
    Union,
)

from getstream.video.rtc import PcmData
from vision_agents.core.edge.types import Participant
from vision_agents.core.events.manager import EventManager
from vision_agents.core.observability import MetricsCollector
from vision_agents.core.base import Component

from . import events
from .events import (
    TTSConnectedEvent,
    TTSDisconnectedEvent,
    TTSErrorEvent,
    TTSSynthesisCompleteEvent,
    TTSSynthesisStartEvent,
)

logger = logging.getLogger(__name__)


@dataclass
class TTSInput:
    text: str
    # True: ``text`` is a chunk of a larger streaming utterance (append/buffer).
    # False: ``text`` is a complete utterance to synthesize as-is.
    delta: bool


@dataclass
class TTSInputEnd:
    """Control message: the current utterance is finished."""


@dataclass
class TTSOutputChunk:
    data: PcmData | None = None  # can be None if it's a final chunk
    index: int = 0
    final: bool = False
    text: str = ""
    synthesis_id: str | None = None


@dataclass
class TTSOutputEnd:
    """Sentinel marking end of TTS output; ``interrupted`` set on barge-in."""

    interrupted: bool = False


class TTS(Component):
    """
    Text-to-Speech base class.

    This abstract class provides the interface for text-to-speech implementations.
    It handles:
    - Converting text to speech
    - Emitting audio events

    Events:
        - audio: Emitted when an audio chunk is available.
            Args: audio_data (bytes), user_metadata (dict)
        - error: Emitted when an error occurs during speech synthesis.
            Args: error (Exception)

    Implementations should inherit from this class and implement the synthesize method.
    """

    # True if the plugin accepts partial text deltas (e.g. token-by-token
    # via a persistent connection). False means it only synthesizes complete
    # utterances. Callers use this to decide whether to feed partial text
    # as it arrives or wait for a full sentence.
    streaming: bool = False

    def __init__(self, provider_name: Optional[str] = None):
        """
        Initialize the TTS base class.

        Args:
            provider_name: Name of the TTS provider (e.g., "cartesia", "elevenlabs")
        """
        super().__init__()
        self.session_id = str(uuid.uuid4())
        self.provider_name = provider_name or self.__class__.__name__
        self.events = EventManager()
        self.events.register_events_from_module(events, ignore_not_compatible=True)
        self.metrics = MetricsCollector()

        # Monotonic epoch counter; incremented on interrupt so stale events
        # emitted before the interrupt can be identified and dropped.
        self._epoch: int = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    async def interrupt(self) -> None:
        """Increment epoch and stop audio. Stale events will be discarded."""
        self._epoch += 1
        await self.stop_audio()

    def _on_connected(self) -> None:
        """Emit TTSConnectedEvent."""
        self.events.send(TTSConnectedEvent(plugin_name=self.provider_name))

    def _on_disconnected(
        self, reason: Optional[str] = None, clean: bool = True
    ) -> None:
        """Emit TTSDisconnectedEvent."""
        self.events.send(
            TTSDisconnectedEvent(
                plugin_name=self.provider_name, reason=reason, clean=clean
            )
        )

    def _emit_error_event(self, error: Exception, *, context: str = "") -> None:
        """Record metric and emit TTSErrorEvent. Caller may also re-raise."""
        self.metrics.on_tts_error(
            provider=self.provider_name,
            error_type=type(error).__name__,
        )
        self.events.send(
            TTSErrorEvent(
                plugin_name=self.provider_name,
                error=error,
                context=context,
            )
        )

    async def _iter_pcm(self, resp: Any) -> AsyncGenerator[PcmData, None]:
        """Yield PcmData chunks from a provider response of various shapes."""
        # Single buffer or PcmData
        if isinstance(resp, (PcmData,)):
            yield resp
            return
        # Async iterable
        if hasattr(resp, "__aiter__"):
            async for item in resp:
                if not isinstance(item, PcmData):
                    raise TypeError(
                        "stream_audio must yield PcmData; wrap provider bytes via PcmData.from_response in the plugin"
                    )
                yield item
            return
        # Sync iterable
        if hasattr(resp, "__iter__") and not isinstance(
            resp, (bytes, bytearray, memoryview, str)
        ):
            for item in resp:
                if not isinstance(item, PcmData):
                    raise TypeError(
                        "stream_audio must yield PcmData; wrap provider bytes via PcmData.from_response in the plugin"
                    )
                yield item
            return
        raise TypeError(f"Unsupported return type from stream_audio: {type(resp)}")

    @abc.abstractmethod
    async def stream_audio(
        self, text: str, *args, **kwargs
    ) -> Union[
        bytes,
        Iterator[bytes],
        AsyncIterator[bytes],
        PcmData,
        Iterator[PcmData],
        AsyncIterator[PcmData],
    ]:
        """
        Convert text to speech audio data.

        This method must be implemented by subclasses.

        Args:
            text: The text to convert to speech
            *args: Additional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Audio data as bytes, an iterator of audio chunks, or an async iterator of audio chunks
        """
        pass

    @abc.abstractmethod
    async def stop_audio(self) -> None:
        """
        Clears the queue and stops playing audio.
        This method can be used manually or under the hood in response to turn events.

        This method must be implemented by subclasses.


        Returns:
            None
        """
        pass

    async def send_iter(
        self,
        text: str,
        participant: Optional[Participant] = None,
        *args,
        **kwargs,
    ) -> AsyncIterator[TTSOutputChunk]:
        """
        Convert text to speech and emit audio events with the desired format.

        Args:
            text: The text to convert to speech
            participant: Optional participant to associate with the audio event
            *args: Additional arguments passed to stream_audio()
            **kwargs: Additional keyword arguments passed to stream_audio()
        """

        start_time = time.perf_counter()
        synthesis_id = str(uuid.uuid4())
        # Store epoch into a variable before yielding
        epoch = self._epoch

        self.events.send(
            TTSSynthesisStartEvent(
                session_id=self.session_id,
                plugin_name=self.provider_name,
                text=text,
                synthesis_id=synthesis_id,
                participant=participant,
            )
        )

        try:
            # Synthesize audio in provider-native format
            response = await self.stream_audio(text, *args, **kwargs)
            if epoch != self._epoch:
                return

            # Calculate synthesis setup time
            total_audio_bytes = 0
            total_audio_ms = 0.0
            chunk_index = 0

            # Fast-path: single buffer -> mark final
            synthesis_time = time.perf_counter() - start_time
            if isinstance(response, PcmData):
                bytes_len, duration_ms = len(response.samples), response.duration_ms
                yield TTSOutputChunk(
                    data=response,
                    index=0,
                    final=True,
                    synthesis_id=synthesis_id,
                    text=text,
                )
                total_audio_bytes += bytes_len
                total_audio_ms += duration_ms
                chunk_index = 1
            else:
                async for pcm in self._iter_pcm(response):
                    if epoch != self._epoch:
                        return
                    # Register the synthesis time only when we get the first chunk
                    if chunk_index == 0:
                        synthesis_time = time.perf_counter() - start_time

                    bytes_len, duration_ms = len(pcm.samples), pcm.duration_ms
                    yield TTSOutputChunk(
                        data=pcm,
                        index=chunk_index,
                        final=False,
                        synthesis_id=synthesis_id,
                        text=text,
                    )
                    total_audio_bytes += bytes_len
                    total_audio_ms += duration_ms
                    chunk_index += 1

                # Emit an empty chunk with "final=True" after the iterator completes.
                # The consumers of these events may use it as a signal
                # to e.g. flush the buffer.
                if chunk_index > 0:
                    yield TTSOutputChunk(
                        data=None,
                        index=chunk_index,
                        final=True,
                        synthesis_id=synthesis_id,
                        text=text,
                    )

            # Use accumulated PcmData duration for total audio duration
            estimated_audio_duration_ms = total_audio_ms

            real_time_factor = (
                (synthesis_time * 1000) / estimated_audio_duration_ms
                if estimated_audio_duration_ms > 0
                else None
            )
            self.events.send(
                TTSSynthesisCompleteEvent(
                    session_id=self.session_id,
                    plugin_name=self.provider_name,
                    synthesis_id=synthesis_id,
                    text=text,
                    participant=participant,
                    total_audio_bytes=total_audio_bytes,
                    synthesis_time_ms=synthesis_time * 1000,
                    audio_duration_ms=estimated_audio_duration_ms,
                    chunk_count=chunk_index,
                    real_time_factor=real_time_factor,
                )
            )
            self.metrics.on_tts_synthesis(
                provider=self.provider_name,
                synthesis_time_ms=synthesis_time * 1000,
                audio_duration_ms=estimated_audio_duration_ms,
                character_count=len(text),
            )
        except Exception as e:
            self._emit_error_event(e, context="synthesis")
            raise

    async def close(self):
        """Close the TTS service and release any resources."""
        # Bump the epoch to allow existing iterators to stop
        self._epoch += 1
        # Clear the buffers and close connections
        await self.stop_audio()
