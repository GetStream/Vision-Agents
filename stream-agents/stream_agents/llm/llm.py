import functools
from typing import List, ParamSpec, TypeVar, Optional, Any, Callable, Protocol, overload, Awaitable, Generic
import os

import anthropic
from anthropic import AsyncAnthropic
from anthropic.resources import AsyncMessages
from anthropic.types import Message
from google import genai
from openai import OpenAI

from stream_agents.agents import Conversation
from stream_agents.processors import BaseProcessor

class LLM:
    def create_response(self, text, processors: List[BaseProcessor]):
        pass

class LLMResponse:
    def __init__(self, original):
        self.original = original

class ClaudeResponse(LLMResponse):
    original : Message


class OpenAILLM(LLM):
    '''
    Use the conversation feature
    '''
    def __init__(self, model: str, api_key: Optional[str] = None, client: Optional[OpenAI] = None):
        self.model = model

        if client is not None:
            self.client = client
        else:
            # If no api_key provided, AsyncAnthropic will look for ANTHROPIC_API_KEY env var
            self.client = OpenAI()

    async def create_response(self, *args, **kwargs) -> ClaudeResponse:
        # TODO: both create response and simple response need to know about history/conversation
        # TODO: both need to know about the "instructions"
        # TODO: upon the first message we could create the conversation? or on init?
        if "model" not in kwargs:
            kwargs["model"] = self.model

        response = self.client.responses.create(
            instructions="You are a coding assistant that talks like a pirate.",
            input="How do I check if a Python object is an instance of a class?",
        )
        # TODO: update message history here with response
        return ClaudeResponse(response)

    async def simple_response(self, text: str, processors: Optional[List[BaseProcessor]] = None, conversation: Conversation =None):
        return await self.create_response(
            input=text
        )


class ClaudeLLM(LLM):
    '''
    Manually keep history
    '''
    def __init__(self, model: str, api_key: Optional[str] = None, client: Optional[AsyncAnthropic] = None):
        self.model = model

        if client is not None:
            self.client = client
        else:
            # If no api_key provided, AsyncAnthropic will look for ANTHROPIC_API_KEY env var
            self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def create_message(self, *args, **kwargs) -> ClaudeResponse:
        # TODO: store message history here
        if "model" not in kwargs:
            kwargs["model"] = self.model

        original = await self.client.messages.create(*args, **kwargs)
        # TODO: update message history here with response
        return ClaudeResponse(original)

    async def simple_response(self, text: str, processors: Optional[List[BaseProcessor]] = None):
        return await self.create_message(
            input=text
        )

class GeminiLLM(LLM):
    '''
    Use the SDK to keep history. (which is partially manual)
    '''
    client: genai.Client

    def __init__(self, model: str, api_key: Optional[str] = None, client: Optional[genai.Client] = None):
        self.model = model

        if client is not None:
            self.client = client
        else:
            self.client = genai.Client()

    async def simple_response(self, text: str, processors: Optional[List[BaseProcessor]] = None):
        return await self.generate_content(
            contents=text
        )
    # basically wrap the Gemini native endpoint
    def generate_content(self, *args, **kwargs):
        if "model" not in kwargs:
            kwargs["model"] = self.model

        client = genai.Client()
        chat = client.chats.create(model="gemini-2.5-flash")

        response = chat.send_message("I have 2 dogs in my house.")

        response = self.client.generate_content(*args, **kwargs)

        return LLMResponse(response)

