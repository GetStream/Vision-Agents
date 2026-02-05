import enum
from dataclasses import dataclass
from typing import Any, Optional

from pyee.asyncio import AsyncIOEventEmitter


@dataclass
class User:
    id: Optional[str] = ""
    name: Optional[str] = ""
    image: Optional[str] = ""


@dataclass
class Participant:
    original: Any
    user_id: str


class TrackType(enum.IntEnum):
    UNSPECIFIED = 0
    AUDIO = 1
    VIDEO = 2
    SCREEN_SHARE = 3
    SCREEN_SHARE_AUDIO = 4


class Connection(AsyncIOEventEmitter):
    """
    To standardize we need to have a method to close
    and a way to receive a callback when the call is ended
    In the future we might want to forward more events
    """

    async def close(self):
        pass
