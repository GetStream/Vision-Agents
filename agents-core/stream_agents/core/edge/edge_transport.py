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
    To normalize

    - join method
    - call/room object
    - open demo/ browser
    """

    def open_demo(self, *args, **kwargs):
        pass



