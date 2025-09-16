#from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, NamedTuple

import numpy as np
from numpy._typing import NDArray
from pyee.asyncio import AsyncIOEventEmitter



@dataclass
class User:
    id: str
    name: str
    image: Optional[str] = ""


@dataclass
class Participant:
    original: Any
    user_id: str


from enum import IntEnum

class TrackType(IntEnum):
    TRACK_TYPE_UNSPECIFIED     = 0
    TRACK_TYPE_AUDIO           = 1
    TRACK_TYPE_VIDEO           = 2
    TRACK_TYPE_SCREEN_SHARE    = 3
    TRACK_TYPE_SCREEN_SHARE_AUDIO = 4

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