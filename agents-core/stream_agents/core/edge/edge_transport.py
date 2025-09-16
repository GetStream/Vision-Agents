"""
Abstraction for stream vs other services here
"""
import abc

from typing import TYPE_CHECKING
from pyee.asyncio import AsyncIOEventEmitter

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

    def open_demo(self, *args, **kwargs):
        pass



