import asyncio
import datetime
import tempfile
from dataclasses import dataclass, asdict
from typing import Optional

import aiortc.mediastreams

from ..edge.types import Participant
from ..llm.events import LLMResponseCompletedEvent
from ..utils.video_forwarder import VideoForwarder


@dataclass
class AgentOptions:
    model_dir: str

    def update(self, other: "AgentOptions") -> "AgentOptions":
        merged_dict = asdict(self)

        for key, value in asdict(other).items():
            if value is not None:
                merged_dict[key] = value

        return AgentOptions(**merged_dict)


# Cache tempdir at module load time to avoid blocking I/O during async operations
_DEFAULT_MODEL_DIR = tempfile.gettempdir()


def default_agent_options():
    return AgentOptions(model_dir=_DEFAULT_MODEL_DIR)


@dataclass
class TrackInfo:
    id: str
    type: int
    processor: str
    priority: int  # higher goes first
    participant: Optional[Participant]
    track: aiortc.mediastreams.VideoStreamTrack
    forwarder: VideoForwarder


@dataclass
class LLMTurn:
    input: str
    participant: Optional[Participant]
    started_at: datetime.datetime
    finished_at: Optional[datetime.datetime] = None
    canceled_at: Optional[datetime.datetime] = None
    response: Optional[LLMResponseCompletedEvent] = None
    task: Optional[asyncio.Task] = None
    turn_finished: bool = False
