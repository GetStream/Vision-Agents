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

    PcmData is the standard audio format container used throughout Vision Agents
    for passing audio between components (agents, transport layers, utilities).

    Audio Format Components:
        - sample_rate: Determines audio quality and frequency range
          * 16000 Hz: Standard for voice/ASR (captures 0-8kHz, human voice range)
          * 24000 Hz: High-quality voice
          * 48000 Hz: CD-quality, suitable for music and multimedia
        - channels: Number of independent audio signals
          * 1 (mono): Single channel, typical for voice applications
          * 2 (stereo): Dual channel, provides spatial audio
        - bit_depth: Dynamic range and precision of audio samples
          * 16-bit: Standard quality (96 dB dynamic range)
          * 24-bit: Professional quality
          * 32-bit: Highest precision
        - data: Raw PCM samples as bytes (interleaved for multi-channel)
        - timestamp: Optional Unix timestamp for synchronization

    Attributes:
        data: Raw PCM audio data as bytes. For multi-channel audio, samples are
              interleaved (e.g., stereo: [L, R, L, R, ...])
        sample_rate: Audio sampling rate in Hz (e.g., 16000, 24000, 48000)
        channels: Number of audio channels (1 for mono, 2 for stereo)
        bit_depth: Bits per sample (default: 16). Common values are 8, 16, 24, 32
        timestamp: Optional timestamp in seconds since epoch when the audio was captured

    Example - Creating PcmData:
        >>> import numpy as np
        >>> import struct
        >>>
        >>> # Generate 1 second of 440 Hz sine wave (A note) at 16kHz mono
        >>> sample_rate = 16000
        >>> duration = 1.0
        >>> frequency = 440.0
        >>> t = np.linspace(0, duration, int(sample_rate * duration))
        >>> samples = np.sin(2 * np.pi * frequency * t)
        >>>
        >>> # Convert to 16-bit PCM
        >>> pcm_samples = (samples * 32767).astype(np.int16)
        >>> audio_data = pcm_samples.tobytes()
        >>>
        >>> # Create PcmData
        >>> pcm = PcmData(
        ...     data=audio_data,
        ...     sample_rate=16000,
        ...     channels=1,
        ...     bit_depth=16,
        ...     timestamp=1234567890.0
        ... )

    Example - Receiving PcmData in callback:
        >>> from vision_agents.core.types import PcmData
        >>>
        >>> def on_audio(pcm_data: PcmData):
        ...     '''Process received audio data.'''
        ...     print(f"Format: {pcm_data.sample_rate} Hz, {pcm_data.channels} ch")
        ...     print(f"Bit depth: {pcm_data.bit_depth}-bit")
        ...     print(f"Data size: {len(pcm_data.data)} bytes")
        ...
        ...     # Calculate duration
        ...     bytes_per_sample = pcm_data.bit_depth // 8
        ...     total_samples = len(pcm_data.data) // (bytes_per_sample * pcm_data.channels)
        ...     duration = total_samples / pcm_data.sample_rate
        ...     print(f"Duration: {duration:.2f} seconds")
        ...
        >>> # Subscribe to audio track
        >>> edge.add_track_subscriber("audio", on_audio)

    Example - Format Validation:
        >>> def validate_audio_format(pcm: PcmData) -> bool:
        ...     '''Validate PcmData has supported format.'''
        ...     if pcm.sample_rate not in [8000, 16000, 24000, 48000]:
        ...         raise ValueError(f"Unsupported sample rate: {pcm.sample_rate}")
        ...     if pcm.channels not in [1, 2]:
        ...         raise ValueError(f"Unsupported channels: {pcm.channels}")
        ...     if pcm.bit_depth not in [8, 16, 24, 32]:
        ...         raise ValueError(f"Unsupported bit depth: {pcm.bit_depth}")
        ...     return True

    See Also:
        - AUDIO_DOCUMENTATION.md: Comprehensive audio configuration guide
        - EdgeTransport.add_track_subscriber(): Subscribe to audio tracks
        - AudioOutputTrack: Playback audio with automatic format conversion
    """

    data: bytes
    sample_rate: int
    channels: int
    bit_depth: int = 16
    timestamp: Optional[float] = None
