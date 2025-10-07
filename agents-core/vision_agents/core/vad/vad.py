import abc
import logging
import time
import uuid
from typing import Optional, Dict, Any, Union

import numpy as np

from getstream.video.rtc.track_util import PcmData
from vision_agents.core.events.manager import EventManager
from getstream.audio.pcm_utils import pcm_to_numpy_array, numpy_array_to_bytes

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
        frame_size: int = 512,
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

        self.sample_rate = sample_rate
        self.frame_size = frame_size
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
        self.speech_buffer = (
            bytearray()
        )  # Use bytearray instead of list of numpy arrays
        self.silence_counter = 0
        self.is_speech_active = False
        self.total_speech_frames = 0
        self.partial_counter = 0
        self._leftover: np.ndarray = np.empty(0, np.int16)
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

        if pcm_data.sample_rate != self.sample_rate:
            raise TypeError(
                f"vad is initialized with sample rate {self.sample_rate} but pcm data has sample rate {pcm_data.sample_rate}"
            )

        # Convert samples to numpy array using shared utility
        samples = pcm_to_numpy_array(pcm_data)
        pcm_data = PcmData(
            samples=samples,
            sample_rate=pcm_data.sample_rate,
            format=pcm_data.format,
        )

        # Prepend leftover samples from previous call
        if len(self._leftover) > 0:
            buffer = np.concatenate([self._leftover, pcm_data.samples])
        else:
            buffer = pcm_data.samples

        # Process audio in full frames only, without zero-padding
        frame_start = 0
        while frame_start + self.frame_size <= len(buffer):
            frame = buffer[frame_start : frame_start + self.frame_size]
            frame_start += self.frame_size

            await self._process_frame(
                PcmData(
                    samples=frame,
                    sample_rate=pcm_data.sample_rate,
                    format=pcm_data.format,
                ),
                participant,
            )

        # Store any remaining samples for the next call
        if frame_start < len(buffer):
            self._leftover = buffer[frame_start:]
        else:
            self._leftover = np.empty(0, np.int16)

        if len(self._leftover) > 0:
            logger.debug(f"Keeping {len(self._leftover)} samples for next processing")

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
            # Append the frame bytes to the bytearray buffer using shared utility
            # Make a copy of samples to avoid BufferError due to memory view restrictions
            frame_bytes = numpy_array_to_bytes(frame.samples)
            self.speech_buffer.extend(frame_bytes)
            self.total_speech_frames += 1
            self.partial_counter += 1

            # Emit partial event every N frames while in speech
            if self.partial_counter >= self.partial_frames:
                # Create a copy of the current speech data
                current_samples = np.frombuffer(
                    self.speech_buffer, dtype=np.int16
                ).copy()
                current_bytes = numpy_array_to_bytes(current_samples)

                # Calculate current duration
                current_duration_ms = (len(current_samples) / self.sample_rate) * 1000

                # Emit structured partial event
                self.events.send(
                    VADPartialEvent(
                        session_id=self.session_id,
                        plugin_name=self.provider_name,
                        audio_data=current_bytes,
                        duration_ms=current_duration_ms,
                        frame_count=len(current_samples) // self.frame_size,
                        user_metadata=participant,
                    )
                )

                logger.debug(
                    f"Emitted partial event with {len(current_samples)} samples"
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
                    user_metadata=participant,
                )
            )

            # Add this frame to the buffer using shared utility
            frame_bytes = numpy_array_to_bytes(frame.samples)
            self.speech_buffer.extend(frame_bytes)

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

        # Convert bytearray to numpy array
        speech_data = np.frombuffer(self.speech_buffer, dtype=np.int16).copy()
        speech_bytes = numpy_array_to_bytes(speech_data)

        # Calculate speech duration
        speech_duration_ms = (len(speech_data) / self.sample_rate) * 1000

        if len(speech_data) >= min_speech_frames * self.frame_size:
            # Emit structured audio event
            self.events.send(
                VADAudioEvent(
                    session_id=self.session_id,
                    plugin_name=self.provider_name,
                    audio_data=speech_bytes,
                    duration_ms=speech_duration_ms,
                    frame_count=len(speech_data) // self.frame_size,
                    user_metadata=user,
                )
            )

            logger.debug(f"Emitted audio event with {len(speech_data)} samples")

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
                    user_metadata=user,
                )
            )

        # Reset state variables
        self.speech_buffer = bytearray()
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
        self.speech_buffer = bytearray()
        self.silence_counter = 0
        self.is_speech_active = False
        self.total_speech_frames = 0
        self.partial_counter = 0
        self._leftover = np.empty(0, np.int16)
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
                user_metadata=user_metadata,
                frame_data_available=len(self.speech_buffer) > 0,
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
