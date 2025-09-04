from typing import List, Optional, Any, TypeVar
import abc
import uuid
from typing import Dict

from pyee.asyncio import AsyncIOEventEmitter

from stream_agents.processors import BaseProcessor

from .event_utils import register_global_event
from .events import (
    LLMErrorEvent,
    LLMRequestEvent,
    LLMResponseEvent,
    PluginClosedEvent,
    PluginInitializedEvent,
)
from ..agents.conversation import Conversation
import anthropic
from anthropic import AsyncAnthropic
from anthropic.types import Message
from google import genai
from openai import OpenAI


class LLMResponse:
    def __init__(self, original, text: str):
        self.original = original
        self.text = text


class LLM(
    AsyncIOEventEmitter,
    abc.ABC,
):
    """Base class for multimodal LLM providers.

    Responsibilities:
    - Normalize requests (messages/options when needed)
    - Emit structured events (request, response, error)
    - Provide a simple non-streaming API via abstract provider hook
    """

    def __init__(
        self,
        provider_name: str,
        model: str,
        client: Optional[TypeVar("C")] = None,
        system_prompt: Optional[str] = None,
    ):
        super().__init__()
        self.session_id = str(uuid.uuid4())
        self.provider_name = provider_name or self.__class__.__name__
        self.model = model
        self.client: Optional[TypeVar("C")] = client
        self.system_prompt = system_prompt

        # Emit initialization event
        init_event = PluginInitializedEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            plugin_type="LLM",
            provider=self.provider_name,
            configuration={"model": self.model} if self.model else None,
        )
        register_global_event(init_event)
        self.emit("initialized", init_event)

        # Default: not realtime/STS
        self.sts: bool = False

    async def create_response(
        self,
        text: str,
        processors: Optional[List[BaseProcessor]] = None,
    ) -> LLMResponse:
        """The general response to be used by an Agent.

        Request a non-streaming response.

        Emits LLM_REQUEST and then either LLM_RESPONSE or LLM_ERROR.
        """
        req_event = LLMRequestEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            model_name=self.model,
            messages=text,
        )
        register_global_event(req_event)
        self.emit("request", req_event)

        try:
            result_text = await self._create_response_impl(
                text=text,
                processors=processors,
            )

            resp_event = LLMResponseEvent(
                session_id=self.session_id,
                plugin_name=self.provider_name,
                normalized_response=None,
                raw=result_text,
            )
            register_global_event(resp_event)
            self.emit("response", resp_event)

            return result_text
        except Exception as e:
            err_event = LLMErrorEvent(
                session_id=self.session_id,
                plugin_name=self.provider_name,
                error=e,
                context="create_response",
            )
            register_global_event(err_event)
            self.emit("error", err_event)
            raise

    @abc.abstractmethod
    async def _create_response_impl(
        self,
        *,
        text: str,
        processors: Optional[List[BaseProcessor]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Provider-specific non-streaming implementation returning plain text."""
        ...

    async def close(self):
        """Close the LLM and release any resources."""
        close_event = PluginClosedEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            plugin_type="LLM",
            provider=self.provider_name,
            cleanup_successful=True,
        )
        register_global_event(close_event)
        self.emit("closed", close_event)


class ClaudeResponse(LLMResponse):
    original: Message


class OpenAILLM(LLM):
    """
    The goal is to standardize the minimal feature set thats needed for the agent integration
    That means

    - sharing instructions
    - keeping conversation history
    - response normalization

    Other than that we aim to give access to the native API methods from openAI as much as possible
    """

    def __init__(
        self, model: str, api_key: Optional[str] = None, client: Optional[OpenAI] = None
    ):
        self.model = model
        self.openai_conversation = None

        if client is not None:
            self.client = client
        else:
            # If no api_key provided, AsyncAnthropic will look for ANTHROPIC_API_KEY env var
            self.client = OpenAI()

    async def create_response(self, *args, **kwargs) -> LLMResponse:
        if "model" not in kwargs:
            kwargs["model"] = self.model

        response = self.client.responses.create(*args, **kwargs)
        # TODO: do we have the response or a standardized response here?
        return LLMResponse(response, response.output_text)

    async def simple_response(
        self,
        text: str,
        processors: Optional[List[BaseProcessor]] = None,
        conversation: "Conversation" = None,
    ):
        if not self.openai_conversation:
            self.openai_conversation = self.client.conversations.create()
        return await self.create_response(
            input=text,
            instructions=conversation.instructions,
            conversation=self.openai_conversation.id,
        )


class ClaudeLLM(LLM):
    """
    Manually keep history
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        client: Optional[AsyncAnthropic] = None,
    ):
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

    async def simple_response(
        self, text: str, processors: Optional[List[BaseProcessor]] = None
    ):
        return await self.create_message(input=text)


class GeminiLLM(LLM):
    """
    Use the SDK to keep history. (which is partially manual)
    """

    client: genai.Client

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        client: Optional[genai.Client] = None,
    ):
        self.model = model

        if client is not None:
            self.client = client
        else:
            self.client = genai.Client()

    async def simple_response(
        self, text: str, processors: Optional[List[BaseProcessor]] = None
    ):
        return await self.generate_content(contents=text)

    # basically wrap the Gemini native endpoint
    def generate_content(self, *args, **kwargs):
        if "model" not in kwargs:
            kwargs["model"] = self.model

        client = genai.Client()
        chat = client.chats.create(model="gemini-2.5-flash")

        response = chat.send_message("I have 2 dogs in my house.")

        response = self.client.generate_content(*args, **kwargs)

        return LLMResponse(response)
