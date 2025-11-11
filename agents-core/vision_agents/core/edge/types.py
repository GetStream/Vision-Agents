from dataclasses import dataclass
from typing import (
    Any,
    Optional,
    Protocol,
    runtime_checkable,
)


from getstream.video.rtc import PcmData
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


@runtime_checkable
class OutputAudioTrack(Protocol):
    """
    A protocol describing an output audio track, the actual implementation depends on the edge transported used
    eg. getstream.video.rtc.audio_track.AudioStreamTrack
    """

    async def write(self, data: PcmData) -> None: ...

    def stop(self) -> None: ...

    async def flush(self) -> None: ...
