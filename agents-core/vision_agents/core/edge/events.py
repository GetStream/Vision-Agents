from dataclasses import dataclass, field
from typing import Optional

from getstream.video.rtc.track_util import PcmData
from vision_agents.core.events import PluginBaseEvent

from .call import Call
from .types import Participant, TrackType


@dataclass
class AudioReceivedEvent(PluginBaseEvent):
    """Event emitted when audio is received from a participant."""

    type: str = field(default="plugin.edge.audio_received", init=False)
    pcm_data: Optional[PcmData] = None


@dataclass
class TrackAddedEvent(PluginBaseEvent):
    """Event emitted when a track is added to the call."""

    type: str = field(default="plugin.edge.track_added", init=False)
    track_id: Optional[str] = None
    track_type: Optional[TrackType] = None


@dataclass
class TrackRemovedEvent(PluginBaseEvent):
    """Event emitted when a track is removed from the call."""

    type: str = field(default="plugin.edge.track_removed", init=False)
    track_id: Optional[str] = None
    track_type: Optional[TrackType] = None


@dataclass
class CallEndedEvent(PluginBaseEvent):
    """Event emitted when a call ends."""

    type: str = field(default="plugin.edge.call_ended", init=False)
    call: Call = field(kw_only=True)


@dataclass
class ParticipantJoinedEvent(PluginBaseEvent):
    """Event emitted when a participant (other than the agent) joins the call."""

    type: str = field(default="plugin.edge.participant_joined", init=False)
    participant: Participant = field(kw_only=True)
    call: Call = field(kw_only=True)


@dataclass
class ParticipantLeftEvent(PluginBaseEvent):
    """Event emitted when a participant (other than the agent) leaves the call."""

    type: str = field(default="plugin.edge.participant_left", init=False)
    participant: Participant = field(kw_only=True)
    call: Call = field(kw_only=True)
