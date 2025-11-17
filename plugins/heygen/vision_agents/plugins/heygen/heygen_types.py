"""Type definitions for HeyGen plugin."""

from enum import Enum


class VideoQuality(str, Enum):
    """Video quality options for HeyGen avatar streaming."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
