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


class Connection(AsyncIOEventEmitter):
    """
    To standardize we need to have a method to close
    and a way to receive a callback when the call is ended
    In the future we might want to forward more events
    """

    async def close(self):
        pass
