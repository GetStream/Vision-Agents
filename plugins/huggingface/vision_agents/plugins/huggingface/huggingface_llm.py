import logging
from typing import Any, Dict, List, Optional

from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant
from huggingface_hub import AsyncInferenceClient
from huggingface_hub.inference._providers import PROVIDER_OR_POLICY_T
from vision_agents.core.llm.llm import LLM, LLMResponseEvent
from vision_agents.core.llm.llm_types import ToolSchema

from . import events
from ._hf_tool_calling import convert_tools_to_hf_format, create_hf_response

logger = logging.getLogger(__name__)


PLUGIN_NAME = "huggingface_llm"


class HuggingFaceLLM(LLM):
    """HuggingFace Inference integration for text-only LLM models.

    This plugin allows developers to interact with models via HuggingFace's
    Inference Providers API. Supports multiple providers (Together, Groq, etc.)
    through a unified interface.

    Features:
        - Streaming responses with real-time chunk events
        - Function/tool calling support
        - Multiple inference provider support

    Examples:

        from vision_agents.plugins import huggingface
        llm = huggingface.LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")

    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        provider: Optional[PROVIDER_OR_POLICY_T] = None,
        base_url: Optional[str] = None,
        client: Optional[AsyncInferenceClient] = None,
    ):
        """Initialize the HuggingFaceLLM class.

        Args:
            model: The HuggingFace model ID to use.
            api_key: HuggingFace API token. Defaults to HF_TOKEN environment variable.
            provider: Inference provider (e.g., "together", "groq", "fireworks-ai"). Defaults to "auto" which auto-selects.
            base_url: Custom API base URL for OpenAI-compatible endpoints (e.g., Baseten).
                Mutually exclusive with provider.
            client: Optional AsyncInferenceClient instance for dependency injection.
        """
        super().__init__()
        self.model = model
        self.provider = provider
        self.events.register_events_from_module(events)

        if base_url and provider:
            raise ValueError("`base_url` and `provider` are mutually exclusive.")

        if client is not None:
            self._client = client
        elif base_url:
            self._client = AsyncInferenceClient(
                base_url=base_url,
                api_key=api_key,
            )
        else:
            self._client = AsyncInferenceClient(
                token=api_key,
                model=model,
                provider=provider,
            )

    async def simple_response(
        self,
        text: str,
        participant: Optional[Participant] = None,
    ) -> LLMResponseEvent[Any]:
        """Create an LLM response from text input.

        This method is called when a new STT transcript is received.

        Args:
            text: The text to respond to.
            participant: The participant object. If not provided, uses "user" role.
        """
        if self._conversation is None:
            logger.warning(
                f'Cannot request a response from the LLM "{self.model}" - '
                "the conversation has not been initialized yet."
            )
            return LLMResponseEvent(original=None, text="")

        if participant is None:
            await self._conversation.send_message(
                role="user", user_id="user", content=text
            )

        messages = await self._build_model_request()
        return await self.create_response(messages=messages)

    async def create_response(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        *,
        input: Optional[Any] = None,
        stream: bool = True,
        **kwargs: Any,
    ) -> LLMResponseEvent:
        """Create a response using HuggingFace's Inference API.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            input: Alternative to messages - will be converted to messages format.
            stream: Whether to stream the response.
            **kwargs: Additional arguments passed to the API.

        Returns:
            LLMResponseEvent with the response.
        """
        if messages is None:
            if input is not None:
                messages = self._input_to_messages(input)
            else:
                messages = await self._build_model_request()

        tools_param = None
        tools_spec = self.get_available_functions()
        if tools_spec:
            tools_param = convert_tools_to_hf_format(tools_spec)

        effective_model = kwargs.pop("model", self.model)
        return await create_hf_response(
            self,
            self._client,
            effective_model,
            PLUGIN_NAME,
            messages,
            tools_param,
            stream,
            **kwargs,
        )

    def _convert_tools_to_provider_format(
        self, tools: List[ToolSchema]
    ) -> List[Dict[str, Any]]:
        return convert_tools_to_hf_format(tools)

    async def _build_model_request(self) -> list[dict]:
        messages: list[dict] = []
        if self._instructions:
            messages.append({"role": "system", "content": self._instructions})

        if self._conversation is not None:
            for message in self._conversation.messages:
                messages.append({"role": message.role, "content": message.content})
        return messages

    def _input_to_messages(self, input_value: Any) -> List[Dict[str, Any]]:
        """Convert input parameter to messages format for API compatibility."""
        messages: List[Dict[str, Any]] = []

        if self._instructions:
            messages.append({"role": "system", "content": self._instructions})

        if isinstance(input_value, str):
            messages.append({"role": "user", "content": input_value})
        elif isinstance(input_value, list):
            for item in input_value:
                if isinstance(item, dict):
                    role = item.get("role", "user")
                    content = item.get("content", "")
                    if item.get("type") == "message":
                        messages.append({"role": role, "content": content})
                    elif item.get("type") == "function_call_output":
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": item.get("call_id", ""),
                                "content": item.get("output", ""),
                            }
                        )
                    else:
                        messages.append({"role": role, "content": content})
                else:
                    messages.append({"role": "user", "content": str(item)})
        else:
            messages.append({"role": "user", "content": str(input_value)})

        return messages

    async def close(self) -> None:
        await self._client.close()
