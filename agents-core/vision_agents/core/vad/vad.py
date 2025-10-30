"""
Base VAD implementation. Avoids external audio utils and relies on PcmData
for serialization where needed.
"""

import abc
import logging
import time
import uuid
from typing import Optional, Dict, Any, Union

import numpy as np

from getstream.video.rtc.track_util import PcmData
from vision_agents.core.events.manager import EventManager
from ..edge.types import Participant

from . import events
from .events import (
    VADPartialEvent,
    VADSpeechStartEvent,
    VADAudioEvent,
    VADSpeechEndEvent,
    VADErrorEvent,
)
from vision_agents.core.events import (
    PluginInitializedEvent,
    PluginClosedEvent,
    AudioFormat,
)

logger = logging.getLogger(__name__)


class VAD(abc.ABC):
    """
    Voice Activity Detection base class.

    This abstract class provides the interface for voice activity detection
    implementations. It handles:
    - Receiving audio data as PCM
    - Detecting speech vs. silence
    - Accumulating speech and discarding silence
    - Flushing accumulated speech when a pause is detected
    - Emitting "audio" events with the speech data
    - Emitting "partial" events while speech is ongoing
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        window_samples: int = 512,
        channels: int = 1,
        audio_format: AudioFormat = AudioFormat.PCM_S16,
        silence_threshold: float = 0.5,
        activation_th: float = 0.5,
        deactivation_th: float = 0.35,
        speech_pad_ms: int = 300,
        min_speech_ms: int = 250,
        max_speech_ms: int = 30000,
        partial_frames: int = 10,
        provider_name: Optional[str] = None,
    ):
        """
        Initialize the VAD.

        Args:
            sample_rate: Audio sample rate in Hz
            frame_size: Size of audio frames to process
            silence_threshold: Threshold for detecting silence (0.0 to 1.0) - deprecated, use activation_th/deactivation_th instead
            activation_th: Threshold for starting speech detection (0.0 to 1.0)
            deactivation_th: Threshold for ending speech detection (0.0 to 1.0)
            speech_pad_ms: Number of milliseconds to pad before/after speech
            min_speech_ms: Minimum milliseconds of speech to emit
            max_speech_ms: Maximum milliseconds of speech before forced flush
            partial_frames: Number of frames to process before emitting a "partial" event
            provider_name: Name of the VAD provider (e.g., "silero")
        """
        super().__init__()

        # Model input spec
        self.sample_rate = int(sample_rate)           # model sample rate (Hz)
        self.channels = int(channels)                 # model channels (1=mono)
        self.audio_format = audio_format              # model PCM format
        self.frame_size = int(window_samples)         # window size at model rate
        # Keep silence_threshold for backward compatibility
        self.silence_threshold = silence_threshold
        self.activation_th = activation_th
        self.deactivation_th = deactivation_th
        self.speech_pad_ms = speech_pad_ms
        self.min_speech_ms = min_speech_ms
        self.max_speech_ms = max_speech_ms
        self.partial_frames = partial_frames
        self.session_id = str(uuid.uuid4())
        self.provider_name = provider_name or self.__class__.__name__
        self.events = EventManager()
        self.events.register_events_from_module(events, ignore_not_compatible=True)

        # State variables
        # Accumulated speech buffer as PcmData (model spec)
        self.speech_buffer: Optional[PcmData] = None
        self.silence_counter = 0
        self.is_speech_active = False
        self.total_speech_frames = 0
        self.partial_counter = 0
        # Rolling buffer of normalized audio at model spec awaiting windowing
        self._model_buffer: Optional[PcmData] = None
        self._speech_start_time: Optional[float] = None

        # Emit initialization event
        self.events.send(
            PluginInitializedEvent(
                session_id=self.session_id,
                plugin_name=self.provider_name,
                plugin_type="VAD",
                provider=self.provider_name,
                configuration={
                    "sample_rate": sample_rate,
                    "activation_threshold": activation_th,
                    "deactivation_threshold": deactivation_th,
                    "min_speech_ms": min_speech_ms,
                    "max_speech_ms": max_speech_ms,
                },
            )
        )

    @abc.abstractmethod
    async def is_speech(self, frame: PcmData) -> float:
        """
        Determine if the audio frame contains speech.

        Args:
            frame: Audio frame data as PcmData

        Returns:
            Probability (0.0 to 1.0) that the frame contains speech
        """
        pass

    async def process_audio(
        self, pcm_data: PcmData, participant: Optional[Participant] = None
    ) -> None:
        """
        Process raw PCM audio data for voice activity detection.

        Args:
            pcm_data: Raw PCM audio data
            user: User metadata to include with emitted audio events
        """

        # Normalize samples to int16 ndarray for processing
        if isinstance(pcm_data.samples, bytes):
            samples = np.frombuffer(pcm_data.samples, dtype=np.int16)
        elif isinstance(pcm_data.samples, np.ndarray):
            samples = (
                pcm_data.samples.astype(np.int16)
                if pcm_data.samples.dtype != np.int16
                else pcm_data.samples
            )
        else:
            raise TypeError(
                f"Unsupported samples type: {type(pcm_data.samples)}; expected bytes or numpy.ndarray"
            )
        incoming = PcmData(samples=samples, sample_rate=pcm_data.sample_rate, format="s16")
        # Resample to model spec
        normalized = incoming.resample(self.sample_rate, self.channels)
        # Append to rolling buffer
        if self._model_buffer is None:
            self._model_buffer = normalized
        else:
            self._model_buffer = self._model_buffer.append(normalized)

        # Consume full windows
        while (
            self._model_buffer is not None
            and isinstance(self._model_buffer.samples, np.ndarray)
            and len(self._model_buffer.samples) >= self.frame_size
        ):
            window = self._model_buffer.samples[: self.frame_size]
            remainder = self._model_buffer.samples[self.frame_size :]
            self._model_buffer = (
                PcmData(samples=remainder, sample_rate=self.sample_rate, format="s16")
                if remainder.size > 0
                else None
            )
            await self._process_frame(
                PcmData(samples=window, sample_rate=self.sample_rate, format="s16"),
                participant,
            )

    async def _process_frame(
        self, frame: PcmData, participant: Optional[Participant] = None
    ) -> None:
        """
        Process a single audio frame.

        Args:
            frame: Audio frame as PcmData
            user: User metadata to include with emitted audio events
        """
        speech_prob = await self.is_speech(frame)

        # Determine speech state based on asymmetric thresholds
        if self.is_speech_active:
            is_speech = (
                speech_prob >= self.deactivation_th
            )  # Continue speech if above deactivation threshold
        else:
            is_speech = (
                speech_prob >= self.activation_th
            )  # Start speech only if above activation threshold

        # Add frame to buffer in all cases during active speech
        if self.is_speech_active:
            # Append frame to the accumulated PcmData buffer
            if self.speech_buffer is None:
                self.speech_buffer = PcmData(
                    samples=frame.samples,
                    sample_rate=frame.sample_rate,
                    format=frame.format,
                )
            else:
                self.speech_buffer = self.speech_buffer.append(frame)
            self.total_speech_frames += 1
            self.partial_counter += 1

            # Emit partial event every N frames while in speech
            if self.partial_counter >= self.partial_frames:
                # Serialize current buffer to bytes for partial event
                if self.speech_buffer is not None:
                    current_bytes = self.speech_buffer.to_bytes()
                    # Estimate sample count for frame_count
                    current_samples_len = (
                        len(self.speech_buffer.samples)
                        if isinstance(self.speech_buffer.samples, np.ndarray)
                        else len(current_bytes) // 2
                    )
                    current_duration_ms = self.speech_buffer.duration_ms
                else:
                    current_bytes = b""
                    current_samples_len = 0
                    current_duration_ms = 0.0

                # Emit structured partial event
                self.events.send(
                    VADPartialEvent(
                        session_id=self.session_id,
                        plugin_name=self.provider_name,
                        audio_data=current_bytes,
                        duration_ms=current_duration_ms,
                        frame_count=current_samples_len // self.frame_size,
                        participant=participant,
                    )
                )

                logger.debug(
                    f"Emitted partial event with {current_samples_len} samples"
                )
                self.partial_counter = 0

            if is_speech:
                # Reset silence counter when speech is detected
                self.silence_counter = 0
            else:
                # Increment silence counter when silence is detected
                self.silence_counter += 1

                # Calculate silence pad frames based on ms
                speech_pad_frames = int(
                    self.speech_pad_ms * self.sample_rate / 1000 / self.frame_size
                )

                # If silence exceeds padding duration, emit audio and reset
                if self.silence_counter >= speech_pad_frames:
                    await self._flush_speech_buffer(participant)

            # Calculate max speech frames based on ms
            max_speech_frames = int(
                self.max_speech_ms * self.sample_rate / 1000 / self.frame_size
            )

            # Force flush if speech duration exceeds maximum
            if self.total_speech_frames >= max_speech_frames:
                await self._flush_speech_buffer(participant)

        # Start collecting speech when detected
        elif is_speech:
            self.is_speech_active = True
            self.silence_counter = 0
            self.total_speech_frames = 1
            self.partial_counter = 1
            self._speech_start_time = time.time()

            # Emit speech start event
            self.events.send(
                VADSpeechStartEvent(
                    session_id=self.session_id,
                    plugin_name=self.provider_name,
                    speech_probability=speech_prob,
                    activation_threshold=self.activation_th,
                    frame_count=1,
                    participant=participant,
                )
            )

            # Initialize the PcmData buffer with this frame
            self.speech_buffer = PcmData(samples=frame.samples, sample_rate=frame.sample_rate, format=frame.format)

    async def _flush_speech_buffer(self, user: Optional[Union[Dict[str, Any], Participant]] = None) -> None:
        """
        Flush the accumulated speech buffer if it meets minimum length requirements.

        Args:
            user: User metadata to include with emitted audio events
        """
        # Calculate min speech frames based on ms
        min_speech_frames = int(
            self.min_speech_ms * self.sample_rate / 1000 / self.frame_size
        )

        # Serialize buffered speech to bytes
        speech_bytes = b""
        speech_samples = 0
        if self.speech_buffer is not None:
            speech_bytes = self.speech_buffer.to_bytes()
            speech_samples = (
                len(self.speech_buffer.samples)
                if isinstance(self.speech_buffer.samples, np.ndarray)
                else len(speech_bytes) // 2
            )

        # Calculate speech duration
        speech_duration_ms = (
            self.speech_buffer.duration_ms if self.speech_buffer is not None else 0.0
        )

        if speech_samples >= min_speech_frames * self.frame_size:
            # Emit structured audio event
            self.events.send(
                VADAudioEvent(
                    session_id=self.session_id,
                    plugin_name=self.provider_name,
                    audio_data=speech_bytes,
                    duration_ms=speech_duration_ms,
                    frame_count=speech_samples // self.frame_size,
                    participant=user,
                )
            )

            logger.debug(f"Emitted audio event with {speech_samples} samples")

        # Emit speech end event if we were actively detecting speech
        if self.is_speech_active and self._speech_start_time:
            total_speech_duration = (time.time() - self._speech_start_time) * 1000
            self.events.send(
                VADSpeechEndEvent(
                    session_id=self.session_id,
                    plugin_name=self.provider_name,
                    speech_probability=0.0,  # Speech has ended
                    deactivation_threshold=self.deactivation_th,
                    total_speech_duration_ms=total_speech_duration,
                    total_frames=self.total_speech_frames,
                    participant=user,
                )
            )

        # Reset state variables
        self.speech_buffer = None
        self.silence_counter = 0
        self.is_speech_active = False
        self.total_speech_frames = 0
        self.partial_counter = 0
        self._speech_start_time = None

    async def flush(self, user: Optional[Dict[str, Any]] = None) -> None:
        """
        Public method to flush any accumulated speech buffer.

        Args:
            user: User metadata to include with emitted audio events
        """
        await self._flush_speech_buffer(user)

    async def reset(self) -> None:
        """Reset the VAD state."""
        self.speech_buffer = None
        self.silence_counter = 0
        self.is_speech_active = False
        self.total_speech_frames = 0
        self.partial_counter = 0
        self._model_buffer = None
        self._speech_start_time = None

    def _emit_error_event(
        self,
        error: Exception,
        context: str = "",
        user_metadata: Optional[Dict[str, Any]] = None,
    ):
        """Emit a structured error event."""
        self.events.send(
            VADErrorEvent(
                session_id=self.session_id,
                plugin_name=self.provider_name,
                error=error,
                context=context,
                participant=user_metadata,
                frame_data_available=(
                    self.speech_buffer is not None
                    and (
                        (isinstance(self.speech_buffer.samples, np.ndarray) and len(self.speech_buffer.samples) > 0)
                        or (isinstance(self.speech_buffer.samples, (bytes, bytearray)) and len(self.speech_buffer.samples) > 0)
                    )
                ),
            )
        )

    async def close(self):
        """Close the VAD service and release any resources."""
        # Flush any remaining speech before closing
        if self.is_speech_active:
            await self.flush()

        # Emit closure event
        self.events.send(
            PluginClosedEvent(
                session_id=self.session_id,
                plugin_name=self.provider_name,
                plugin_type="VAD",
                provider=self.provider_name,
                cleanup_successful=True,
            )
        )
