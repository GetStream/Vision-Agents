"""LocalEdge package - Refactored local RTC edge transport.

This package provides a modular implementation of the LocalEdge transport,
split into focused components for maintainability and testability.
"""

from .config import AudioConfig, GStreamerConfig, LocalEdgeConfig, VideoConfig
from .core import LocalEdge

__all__ = [
    "LocalEdge",
    "LocalEdgeConfig",
    "AudioConfig",
    "VideoConfig",
    "GStreamerConfig",
]
