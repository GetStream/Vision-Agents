import abc
import logging
import time
import uuid
from typing import Optional, Dict, Union, Iterator, AsyncIterator, AsyncGenerator, Any

import av

from vision_agents.core.events.manager import EventManager

from . import events
from .events import (
    TTSAudioEvent,
    TTSSynthesisStartEvent,
    TTSSynthesisCompleteEvent,
    TTSErrorEvent,
)
from vision_agents.core.events import (
    PluginClosedEvent,
    AudioFormat,
)
from ..observability import (
    tts_latency_ms,
    tts_errors,
)
from ..edge.types import PcmData

logger = logging.getLogger(__name__)


class TTS(abc.ABC):
    """
    Text-to-Speech base class.

    This abstract class provides the interface for text-to-speech implementations.
    It handles:
    - Converting text to speech
    - Resampling and rechanneling audio to a desired format
    - Emitting audio events

    Events:
        - audio: Emitted when an audio chunk is available.
            Args: audio_data (bytes), user_metadata (dict)
        - error: Emitted when an error occurs during speech synthesis.
            Args: error (Exception)

    Implementations should inherit from this class and implement the synthesize method.
    """

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

        # Desired output audio format (what downstream audio track expects)
        self._desired_sample_rate: int = 16000
        self._desired_channels: int = 1
        self._desired_format: AudioFormat = AudioFormat.PCM_S16

        # Persistent resampler to avoid discontinuities between chunks
        self._resampler: Optional[av.AudioResampler] = None
        self._resampler_input_rate: Optional[int] = None
        self._resampler_input_channels: Optional[int] = None

    def set_output_format(
        self,
        sample_rate: int,
        channels: int = 1,
        audio_format: AudioFormat = AudioFormat.PCM_S16,
    ) -> None:
        """Set the desired output audio format for emitted events.

        The agent should call this with its output track properties so this
        TTS instance can resample and rechannel audio appropriately.

        Args:
            sample_rate: Desired sample rate in Hz (e.g., 48000)
            channels: Desired channel count (1 for mono, 2 for stereo)
            audio_format: Desired audio format (defaults to PCM S16)
        """
        self._desired_sample_rate = int(sample_rate)
        self._desired_channels = int(channels)
        self._desired_format = audio_format

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

    def _emit_chunk(
        self,
        pcm: PcmData,
        idx: int,
        is_final: bool,
        synthesis_id: str,
        text: str,
        user: Optional[Dict[str, Any]],
    ) -> tuple[int, float]:
        """Emit TTSAudioEvent; return (bytes_len, duration_ms)."""

        # Resample to desired format if needed
        pcm = pcm.resample(
            target_sample_rate=self._desired_sample_rate,
            target_channels=self._desired_channels,
        )

        self.events.send(
            TTSAudioEvent(
                session_id=self.session_id,
                plugin_name=self.provider_name,
                data=pcm,
                synthesis_id=synthesis_id,
                text_source=text,
                participant=user,
                chunk_index=idx,
                is_final_chunk=is_final,
            )
        )
        return len(pcm.samples), pcm.duration_ms

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

    async def send(
        self, text: str, user: Optional[Dict[str, Any]] = None, *args, **kwargs
    ):
        """
        Convert text to speech and emit audio events with the desired format.

        Args:
            text: The text to convert to speech
            user: Optional user metadata to include with the audio event
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """

        start_time = time.time()
        synthesis_id = str(uuid.uuid4())

        logger.debug(
            "Starting text-to-speech synthesis", extra={"text_length": len(text)}
        )

        self.events.send(
            TTSSynthesisStartEvent(
                session_id=self.session_id,
                plugin_name=self.provider_name,
                text=text,
                synthesis_id=synthesis_id,
                participant=user,
            )
        )

        try:
            # Synthesize audio in provider-native format
            response = await self.stream_audio(text, *args, **kwargs)

            # Calculate synthesis setup time
            synthesis_time = time.time() - start_time

            total_audio_bytes = 0
            total_audio_ms = 0.0
            chunk_index = 0

            # Fast-path: single buffer -> mark final
            if isinstance(response, (PcmData,)):
                bytes_len, dur_ms = self._emit_chunk(
                    response, 0, True, synthesis_id, text, user
                )
                total_audio_bytes += bytes_len
                total_audio_ms += dur_ms
                chunk_index = 1
            else:
                async for pcm in self._iter_pcm(response):
                    bytes_len, dur_ms = self._emit_chunk(
                        pcm, chunk_index, False, synthesis_id, text, user
                    )
                    total_audio_bytes += bytes_len
                    total_audio_ms += dur_ms
                    chunk_index += 1

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
                    participant=user,
                    total_audio_bytes=total_audio_bytes,
                    synthesis_time_ms=synthesis_time * 1000,
                    audio_duration_ms=estimated_audio_duration_ms,
                    chunk_count=chunk_index,
                    real_time_factor=real_time_factor,
                )
            )
        except Exception as e:
            # Metrics: error counter
            tts_errors.add(1, attributes={"tts_class": self.__class__.__name__})
            self.events.send(
                TTSErrorEvent(
                    session_id=self.session_id,
                    plugin_name=self.provider_name,
                    error=e,
                    context="synthesis",
                    text_source=text,
                    synthesis_id=synthesis_id or None,
                    participant=user,
                )
            )
            raise
        finally:
            elapsed_ms = (time.time() - start_time) * 1000.0
            tts_latency_ms.record(
                elapsed_ms, attributes={"tts_class": self.__class__.__name__}
            )

    async def close(self):
        """Close the TTS service and release any resources."""
        self.events.send(
            PluginClosedEvent(
                session_id=self.session_id,
                plugin_name=self.provider_name,
                plugin_type="TTS",
                provider=self.provider_name,
                cleanup_successful=True,
            )
        )
