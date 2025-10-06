#from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Optional, NamedTuple
import logging

import numpy as np
from numpy._typing import NDArray
from pyee.asyncio import AsyncIOEventEmitter
import av

logger = logging.getLogger(__name__)


@dataclass
class User:
    id: str
    name: str
    image: Optional[str] = ""


@dataclass
class Participant:
    original: Any
    user_id: str


class TrackType(StrEnum):
    TRACK_TYPE_UNSPECIFIED     = "unspecified"
    TRACK_TYPE_AUDIO           = "audio"
    TRACK_TYPE_VIDEO           = "video"
    TRACK_TYPE_SCREEN_SHARE    = "screen_share" # TODO: Verify its correct
    TRACK_TYPE_SCREEN_SHARE_AUDIO = "screen_share_audio"

TRACK_TYPE_UNSPECIFIED      = TrackType.TRACK_TYPE_UNSPECIFIED
TRACK_TYPE_AUDIO            = TrackType.TRACK_TYPE_AUDIO
TRACK_TYPE_VIDEO            = TrackType.TRACK_TYPE_VIDEO
TRACK_TYPE_SCREEN_SHARE     = TrackType.TRACK_TYPE_SCREEN_SHARE
TRACK_TYPE_SCREEN_SHARE_AUDIO = TrackType.TRACK_TYPE_SCREEN_SHARE_AUDIO


class Connection(AsyncIOEventEmitter):
    """
    To standardize we need to have a method to close
    and a way to receive a callback when the call is ended
    In the future we might want to forward more events
    """
    async def close(self):
        pass


class PcmData(NamedTuple):
    """
    A named tuple representing PCM audio data.

    Attributes:
        format: The format of the audio data.
        sample_rate: The sample rate of the audio data.
        samples: The audio samples as a numpy array.
        pts: The presentation timestamp of the audio data.
        dts: The decode timestamp of the audio data.
        time_base: The time base for converting timestamps to seconds.
    """

    format: str
    sample_rate: int
    samples: NDArray
    pts: Optional[int] = None  # Presentation timestamp
    dts: Optional[int] = None  # Decode timestamp
    time_base: Optional[float] = None  # Time base for converting timestamps to seconds

    @property
    def duration(self) -> float:
        """
        Calculate the duration of the audio data in seconds.

        Returns:
            float: Duration in seconds.
        """
        # The samples field contains a numpy array of audio samples
        # For s16 format, each element in the array is one sample (int16)
        # For f32 format, each element in the array is one sample (float32)

        if isinstance(self.samples, np.ndarray):
            # Direct count of samples in the numpy array
            num_samples = len(self.samples)
        elif isinstance(self.samples, bytes):
            # If samples is bytes, calculate based on format
            if self.format == "s16":
                # For s16 format, each sample is 2 bytes (16 bits)
                num_samples = len(self.samples) // 2
            elif self.format == "f32":
                # For f32 format, each sample is 4 bytes (32 bits)
                num_samples = len(self.samples) // 4
            else:
                # Default assumption for other formats (treat as raw bytes)
                num_samples = len(self.samples)
        else:
            # Fallback: try to get length
            try:
                num_samples = len(self.samples)
            except TypeError:
                logger.warning(
                    f"Cannot determine sample count for type {type(self.samples)}"
                )
                return 0.0

        # Calculate duration based on sample rate
        return num_samples / self.sample_rate

    @property
    def pts_seconds(self) -> Optional[float]:
        if self.pts is not None and self.time_base is not None:
            return self.pts * self.time_base
        return None

    @property
    def dts_seconds(self) -> Optional[float]:
        if self.dts is not None and self.time_base is not None:
            return self.dts * self.time_base
        return None

    @classmethod
    def from_bytes(
        cls, 
        audio_bytes: bytes, 
        sample_rate: int = 16000, 
        format: str = "s16"
    ) -> "PcmData":
        """
        Create PcmData from raw audio bytes.
        
        Args:
            audio_bytes: Raw audio data as bytes
            sample_rate: Sample rate in Hz
            format: Audio format (e.g., "s16", "f32")
            
        Returns:
            PcmData object
        """
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        return cls(samples=audio_array, sample_rate=sample_rate, format=format)

    def resample(self, target_sample_rate: int) -> "PcmData":
        """
        Resample PcmData to a different sample rate using AV library.
        
        Args:
            target_sample_rate: Target sample rate in Hz
            
        Returns:
            New PcmData object with resampled audio
        """
        if self.sample_rate == target_sample_rate:
            return self
        
        # Ensure samples are 2D for AV library (channels, samples)
        samples = self.samples
        if samples.ndim == 1:
            # Reshape 1D array to 2D (1 channel, samples)
            samples = samples.reshape(1, -1)
        
        # Create AV audio frame from the samples
        frame = av.AudioFrame.from_ndarray(samples, format='s16', layout='mono')
        frame.sample_rate = self.sample_rate
        
        # Create resampler
        resampler = av.AudioResampler(
            format='s16',
            layout='mono',
            rate=target_sample_rate
        )
        
        # Resample the frame
        resampled_frames = resampler.resample(frame)
        if resampled_frames:
            resampled_frame = resampled_frames[0]
            resampled_samples = resampled_frame.to_ndarray()
            
            # AV returns (channels, samples), so for mono we want the first (and only) channel
            if len(resampled_samples.shape) > 1:
                # Take the first channel (mono)
                resampled_samples = resampled_samples[0]
            
            # Convert to int16
            resampled_samples = resampled_samples.astype(np.int16)
            
            return PcmData(
                samples=resampled_samples,
                sample_rate=target_sample_rate,
                format=self.format,
                pts=self.pts,
                dts=self.dts,
                time_base=self.time_base
            )
        else:
            # If resampling failed, return original data
            return self