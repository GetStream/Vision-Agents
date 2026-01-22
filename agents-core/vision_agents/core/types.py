"""Core type definitions for Vision Agents framework.

This module provides framework-native types that are independent of
transport-specific implementations (e.g., GetStream RTC types).
"""

from dataclasses import dataclass
from typing import Optional


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
