import functools
from typing import List, ParamSpec, TypeVar, Optional, Any, Callable, Protocol, overload, Awaitable, Generic
import os

import anthropic
from anthropic import AsyncAnthropic
from anthropic.resources import AsyncMessages
from anthropic.types import Message

from stream_agents.processors import BaseProcessor
P = ParamSpec("P")
T = TypeVar("T")
CreateCall = Callable[P, Awaitable[T]]


def wraps_claude(function: Callable[P, Any]) -> Callable[P, Optional[Any]]:
    def decorator(*args: P.args, **kwargs: P.kwargs) -> Optional[Any]:
        create: CreateCall[P, T] = AsyncMessages.create
        return function(*args, **kwargs)
    return decorator

class LLM:
    def create_response(self, text, processors: List[BaseProcessor]):
        pass

class LLMResponse:
    def __init__(self, original):
        self.original = original

class ClaudeResponse(LLMResponse):
    original : Message

class OpenAILLM(LLM):
    pass

create: CreateCall[P, T] = AsyncMessages.create

class ClaudeLLM(LLM):
    def __init__(self, model: str, api_key: Optional[str] = None, client: Optional[AsyncAnthropic] = None):
        self.model = model

        if client is not None:
            self.client = client
        else:
            # If no api_key provided, AsyncAnthropic will look for ANTHROPIC_API_KEY env var
            self.client = anthropic.AsyncAnthropic(api_key=api_key)

        # TODO: some defaults here like model?
        self.create_message= bind( AsyncMessages.create)


    # Type-preserving wrapper that forwards to client.messages.create
    # This preserves the exact parameter types and return type
    # The signature matches client.messages.create exactly
    async def _create_message(self, *args: P.args, **kwargs: P.kwargs) -> ClaudeResponse:
        # TODO: store message history here
        if "model" not in kwargs:
            kwargs["model"] = self.model

        original = await self.client.messages.create(*args, **kwargs)
        # TODO: update message history here with response
        return ClaudeResponse(original)

    async def create_response(self, text: str, processors: Optional[List[BaseProcessor]] = None):
        messages = [
            {"role": "user", "content": text}
        ]
        return await self.create_message(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages
        )

class GeminiLLM(LLM):
    # standardized/ simplified endpoint
    def create_response(self, text, input_state):
        response = self.client.generate(text)
        self.update_chat_history(text)

        # return the standardized response. keep original response in response.original
        return LLMResponse(response)

    # basically wrap the Gemini native endpoint
    def generate(self, *args, **kwargs):
        text = "hi"
        self.update_chat_history(text)

        response = self.client.generate(*args, **kwargs)

        return LLMResponse(response)

