"""
Silero Voice Activity Detection (VAD) implementation.

This module provides a high-performance speech detection implementation using the
Silero VAD model (https://github.com/snakers4/silero-vad).

Features:
- Asymmetric thresholds for speech detection (activation_th and deactivation_th)
- Early partial events for real-time UI feedback during speech
- Memory-efficient audio buffering using bytearrays
- Automatic resampling to model's required rate (typically 16kHz)
- GPU acceleration support with automatic fallback to CPU
- Optional ONNX runtime support for potential performance improvements

The implementation is designed for streaming audio, handling partial frames
and emitting events when speech is detected.
"""

from .vad import SileroVAD

__all__ = ["SileroVAD"]
