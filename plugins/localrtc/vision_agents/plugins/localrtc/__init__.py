"""Local RTC plugin for Vision Agents."""

from .edge import LocalEdge as Edge
from .room import LocalRoom

__all__ = ["Edge", "LocalRoom"]
