from __future__ import annotations

import abc
from typing import Optional, TYPE_CHECKING

from pyee.asyncio import AsyncIOEventEmitter

if TYPE_CHECKING:
    from stream_agents.core.agents import Agent
    from stream_agents.core.agents.conversation import Conversation


from typing import List, TypeVar, Any, Callable, Generic, Optional as TypingOptional


from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant
from stream_agents.core.processors import BaseProcessor

T = TypeVar("T")


class LLMResponse(Generic[T]):
    def __init__(self, original: T, text: str):
        self.original = original
        self.text = text


BeforeCb = Callable[[List[Any]], None]
AfterCb = Callable[[LLMResponse], None]


class LLM(AsyncIOEventEmitter, abc.ABC):
    # if we want to use realtime/ sts behaviour
    sts: bool = False

    before_response_listener: BeforeCb
    after_response_listener: AfterCb
    agent: Optional["Agent"]
    _conversation: Optional["Conversation"]

    def __init__(self):
        super().__init__()
        self.agent = None

    async def simple_response(
        self,
        text: str,
        processors: TypingOptional[List[BaseProcessor]] = None,
        participant: TypingOptional[Participant] = None,
    ) -> LLMResponse[Any]:
        raise NotImplementedError

    def attach_agent(self, agent: Agent):
        self.agent = agent
        self._conversation = agent.conversation