"""Core type definitions for Vision Agents framework.

This module provides framework-native types that are independent of
transport-specific implementations (e.g., GetStream RTC types).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TrackType(str, Enum):
    """Media track type enumeration.

    This is a framework-native enumeration for representing different types of
    media tracks, decoupled from any specific transport layer implementation.

    Values:
        AUDIO: Audio track type for voice or sound data
        VIDEO: Video track type for camera or visual content
        SCREENSHARE: Screen sharing track type for desktop/window capture
    """

    AUDIO = "audio"
    VIDEO = "video"
    SCREENSHARE = "screenshare"


@dataclass
class PcmData:
    """PCM (Pulse Code Modulation) audio data representation.

    This is a framework-native type for representing raw audio data,
    decoupled from any specific transport layer implementation.

    Attributes:
        data: Raw PCM audio data as bytes
        sample_rate: Audio sampling rate in Hz (e.g., 16000, 24000, 48000)
        channels: Number of audio channels (1 for mono, 2 for stereo)
        bit_depth: Bits per sample (default: 16). Common values are 8, 16, 24, 32
        timestamp: Optional timestamp in seconds since epoch when the audio was captured
    """

    data: bytes
    sample_rate: int
    channels: int
    bit_depth: int = 16
    timestamp: Optional[float] = None
