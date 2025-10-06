from dataclasses import dataclass, field

from vision_agents.core.edge.types import PcmData
from vision_agents.core.events import PluginBaseEvent
from typing import Optional, Any


@dataclass
class AudioReceivedEvent(PluginBaseEvent):
    """Event emitted when audio is received from a participant."""
    type: str = field(default='plugin.edge.audio_received', init=False)
    pcm_data: Optional[PcmData] = None
    participant: Optional[Any] = None


@dataclass
class TrackAddedEvent(PluginBaseEvent):
    """Event emitted when a track is added to the call."""
    type: str = field(default='plugin.edge.track_added', init=False)
    track_id: Optional[str] = None
    track_type: Optional[str] = None
    user: Optional[Any] = None


@dataclass
class TrackEndedEvent(PluginBaseEvent):
    """Event emitted when a track ends."""
    type: str = field(default='plugin.edge.track_ended', init=False)
    track_id: Optional[str] = None
    track_type: Optional[str] = None
    user: Optional[Any] = None


@dataclass
class CallEndedEvent(PluginBaseEvent):
    """Event emitted when a call ends."""
    type: str = field(default='plugin.edge.call_ended', init=False)
    args: Optional[tuple] = None
    kwargs: Optional[dict] = None
