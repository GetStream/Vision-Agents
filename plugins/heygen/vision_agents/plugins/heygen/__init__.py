"""HeyGen avatar plugin for Vision Agents.

This plugin provides HeyGen's interactive avatar streaming capabilities,
allowing AI agents to have realistic avatar video output with lip-sync.
"""

from .heygen_avatar_publisher import AvatarPublisher
from .heygen_types import VideoQuality

__all__ = [
    "AvatarPublisher",
    "VideoQuality",
]
