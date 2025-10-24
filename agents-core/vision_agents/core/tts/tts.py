import abc
import av
import logging
import time
import uuid
from typing import Optional, Dict, Union, Iterator, AsyncIterator, AsyncGenerator, Any

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
    tts_bytes_streamed,
    tts_errors,
    tts_events_emitted,
)
from ..edge.types import PcmData
import numpy as np

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
        # Agent can override via set_output_format
        self._desired_sample_rate: int = 16000
        self._desired_channels: int = 1
        self._desired_format: AudioFormat = AudioFormat.PCM_S16
        # Native/provider audio format default (used only if plugin returns raw bytes)
        self._native_sample_rate: int = 16000
        self._native_channels: int = 1
        self._native_format: AudioFormat = AudioFormat.PCM_S16
        # Persistent resampler to avoid discontinuities between chunks
        self._resampler = None
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

        self._resampler = None
        self._resampler_input_rate = None
        self._resampler_input_channels = None

    def _get_resampler(self, input_rate: int, input_channels: int):
        """Get or create a persistent resampler for the given input format.

        This avoids creating a new resampler for each chunk, which causes
        discontinuities and clicking artifacts in the output audio.

        Args:
            input_rate: Input sample rate
            input_channels: Input channel count

        Returns:
            PyAV AudioResampler instance
        """

        if self._resampler is not None and self._resampler_input_rate == input_rate and self._resampler_input_channels == input_channels:
            return self._resampler

        in_layout = "mono" if input_channels == 1 else "stereo"
        out_layout = "mono" if self._desired_channels == 1 else "stereo"

        self._resampler = av.AudioResampler(
            format="s16", layout=out_layout, rate=self._desired_sample_rate
        )
        self._resampler_input_rate = input_rate
        self._resampler_input_channels = input_channels

        logger.debug(
            "Created persistent resampler: %s@%dHz -> %s@%dHz",
            in_layout,
            input_rate,
            out_layout,
            self._desired_sample_rate,
        )

        return self._resampler

    def _normalize_to_pcm(self, item: Union[bytes, bytearray, PcmData, Any]) -> PcmData:
        """Normalize a chunk to PcmData using the native provider format."""
        if isinstance(item, PcmData):
            return item
        data = getattr(item, "data", item)
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("Chunk is not bytes or PcmData")
        fmt = (
            self._native_format.value
            if hasattr(self._native_format, "value")
            else "s16"
        )
        return PcmData.from_bytes(
            bytes(data),
            sample_rate=self._native_sample_rate,
            channels=self._native_channels,
            format=fmt,
        )

    async def _iter_pcm(self, resp: Any) -> AsyncGenerator[PcmData, None]:
        """Yield PcmData chunks from a provider response of various shapes."""
        # Single buffer or PcmData
        if isinstance(resp, (bytes, bytearray, PcmData)):
            yield self._normalize_to_pcm(resp)
            return
        # Async iterable
        if hasattr(resp, "__aiter__"):
            async for item in resp:
                yield self._normalize_to_pcm(item)
            return
        # Sync iterable (avoid treating bytes-like as iterable of ints)
        if hasattr(resp, "__iter__") and not isinstance(resp, (str, bytes, bytearray)):
            for item in resp:
                yield self._normalize_to_pcm(item)
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
        """Resample, serialize, emit TTSAudioEvent; return (bytes_len, duration_ms)."""

        if (
            pcm.sample_rate == self._desired_sample_rate
            and pcm.channels == self._desired_channels
        ):
            # No resampling needed
            pcm_out = pcm
        else:
            resampler = self._get_resampler(pcm.sample_rate, pcm.channels)

            # Prepare input frame in planar format
            samples = pcm.samples
            if isinstance(samples, np.ndarray):
                if samples.ndim == 1:
                    if pcm.channels > 1:
                        cmaj = np.tile(samples, (pcm.channels, 1))
                    else:
                        cmaj = samples.reshape(1, -1)
                elif samples.ndim == 2:
                    ch = pcm.channels if pcm.channels else 1
                    if samples.shape[0] == ch:
                        cmaj = samples
                    elif samples.shape[1] == ch:
                        cmaj = samples.T
                    else:
                        if samples.shape[1] > samples.shape[0]:
                            cmaj = samples
                        else:
                            cmaj = samples.T
                cmaj = np.ascontiguousarray(cmaj)
            else:
                # Shouldn't happen, but handle it
                cmaj = (
                    samples.reshape(1, -1)
                    if isinstance(samples, np.ndarray)
                    else samples
                )

            in_layout = "mono" if pcm.channels == 1 else "stereo"
            frame = av.AudioFrame.from_ndarray(cmaj, format="s16p", layout=in_layout)
            frame.sample_rate = pcm.sample_rate

            # Resample using persistent resampler
            resampled_frames = resampler.resample(frame)

            if resampled_frames:
                resampled_frame = resampled_frames[0]
                raw_array = resampled_frame.to_ndarray()
                num_frames = resampled_frame.samples

                # Handle PyAV's packed format quirk
                ch = self._desired_channels
                if raw_array.ndim == 2 and raw_array.shape[0] == 1 and ch > 1:
                    flat = raw_array.reshape(-1)
                    if len(flat) == num_frames * ch:
                        resampled_samples = flat.reshape(-1, ch).T
                    else:
                        resampled_samples = flat.reshape(ch, -1)
                elif raw_array.ndim == 2:
                    if raw_array.shape[1] == ch:
                        resampled_samples = raw_array.T
                    elif raw_array.shape[0] == ch:
                        resampled_samples = raw_array
                    else:
                        resampled_samples = raw_array.T
                elif raw_array.ndim == 1:
                    if ch == 1:
                        resampled_samples = raw_array
                    else:
                        resampled_samples = np.tile(raw_array, (ch, 1))
                else:
                    resampled_samples = raw_array.reshape(ch, -1)

                if resampled_samples.dtype != np.int16:
                    resampled_samples = resampled_samples.astype(np.int16)

                pcm_out = PcmData(
                    samples=resampled_samples,
                    sample_rate=self._desired_sample_rate,
                    format="s16",
                    channels=self._desired_channels,
                )
            else:
                # Resampling failed, use original
                pcm_out = pcm

        payload = pcm_out.to_bytes()
        # Metrics: counters per chunk
        attrs = {"tts_class": self.__class__.__name__}
        tts_bytes_streamed.add(len(payload), attributes=attrs)
        tts_events_emitted.add(1, attributes=attrs)
        self.events.send(
            TTSAudioEvent(
                session_id=self.session_id,
                plugin_name=self.provider_name,
                audio_data=payload,
                synthesis_id=synthesis_id,
                text_source=text,
                user_metadata=user,
                chunk_index=idx,
                is_final_chunk=is_final,
                audio_format=self._desired_format,
                sample_rate=self._desired_sample_rate,
                channels=self._desired_channels,
            )
        )
        return len(payload), pcm_out.duration_ms

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

        # Reset resampler for each new synthesis to ensure clean state
        self._resampler = None
        self._resampler_input_rate = None
        self._resampler_input_channels = None

        logger.debug(
            "Starting text-to-speech synthesis", extra={"text_length": len(text)}
        )

        self.events.send(
            TTSSynthesisStartEvent(
                session_id=self.session_id,
                plugin_name=self.provider_name,
                text=text,
                synthesis_id=synthesis_id,
                user_metadata=user,
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
            if isinstance(response, (bytes, bytearray, PcmData)):
                bytes_len, dur_ms = self._emit_chunk(
                    self._normalize_to_pcm(response), 0, True, synthesis_id, text, user
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
                    user_metadata=user,
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
                    user_metadata=user,
                )
            )
            raise
        finally:
            # Metrics: latency histogram for the entire send call
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
