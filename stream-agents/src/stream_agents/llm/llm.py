from typing import List, TypeVar, Optional, Any, Callable, Protocol, overload, Awaitable, Generic

from av.dictionary import Dictionary

from stream_agents.processors import BaseProcessor

T = TypeVar("T")

class LLMResponse(Generic[T]):
    def __init__(self, original: T, text: str):
        self.original = original
        self.text = text

BeforeCb = Callable[[List[Dictionary]], None]
AfterCb  = Callable[[LLMResponse], None]



class LLM:
    # if we want to use realtime/ sts behaviour
    sts: bool = False

    before_response_listener: BeforeCb
    after_response_listener: AfterCb
    agent: Optional["Agent"]


    def __init__(self):
        self.agent = None

    def simple_response(self, text, processors: List[BaseProcessor]) -> LLMResponse[Any]:
        pass

    def attach_agent(self, agent):
        self.agent = agent
        self.before_response_listener = lambda x: agent.add_to_conversation(x)
        self.after_response_listener = lambda x: agent.after_response(x)

    def set_before_response_listener(self, before_response_listener: BeforeCb):
        self.before_response_listener = before_response_listener

    def set_after_response_listener(self, after_response_listener: AfterCb):
        self.after_response_listener = after_response_listener







