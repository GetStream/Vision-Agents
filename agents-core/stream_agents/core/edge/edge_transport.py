"""
Abstraction for stream vs other services here
"""
import abc

from typing import TYPE_CHECKING
from pyee.asyncio import AsyncIOEventEmitter

from stream_agents.core.edge.types import User

if TYPE_CHECKING:

    from stream_agents.core.agents import Agent


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
    def create_audio_track(self):
        pass

    @abc.abstractmethod
    def close(self):
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


