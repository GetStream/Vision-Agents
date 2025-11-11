"""
Abstraction for stream vs other services here
"""

import abc

from typing import TYPE_CHECKING, Any, Optional

import aiortc
from pyee.asyncio import AsyncIOEventEmitter

from vision_agents.core.edge.types import User, OutputAudioTrack

if TYPE_CHECKING:
    pass


class EdgeTransport(AsyncIOEventEmitter, abc.ABC):
    """
    TODO: what's not done yet

    - call type
    - participant type
    - audio track type
    - pcm data type

    """

    @abc.abstractmethod
    async def create_user(self, user: User):
        pass

    @abc.abstractmethod
    def create_audio_track(self) -> OutputAudioTrack:
        pass

    @abc.abstractmethod
    async def close(self):
        pass

    @abc.abstractmethod
    def open_demo(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    async def join(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    async def publish_tracks(self, audio_track, video_track):
        pass

    @abc.abstractmethod
    async def create_conversation(self, call: Any, user: User, instructions):
        pass

    @abc.abstractmethod
    def add_track_subscriber(
        self, track_id: str
    ) -> Optional[aiortc.mediastreams.MediaStreamTrack]:
        pass
