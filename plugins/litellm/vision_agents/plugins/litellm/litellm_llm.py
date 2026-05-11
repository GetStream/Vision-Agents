"""LiteLLM Chat Completions LLM plugin.

Provides a text-in/text-out LLM backed by the litellm SDK,
supporting 100+ providers through a single interface.
"""

import json
import logging
import time

import litellm
from vision_agents.core.llm.events import (
    LLMRequestStartedEvent,
    LLMResponseChunkEvent,
    LLMResponseCompletedEvent,
)
from vision_agents.core.llm.llm import LLM, LLMResponseEvent
from vision_agents.core.llm.llm_types import NormalizedToolCallItem, ToolSchema

logger = logging.getLogger(__name__)

PLUGIN_NAME = "litellm"


class LiteLLMChatCompletions(LLM):
    """LLM plugin that routes to 100+ providers via the litellm SDK.

    Examples:

        from vision_agents.plugins.litellm import LiteLLMChatCompletions

        llm = LiteLLMChatCompletions(model="anthropic/claude-sonnet-4-20250514")
        llm = LiteLLMChatCompletions(model="azure/gpt-4o", api_key="...")
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        api_key: str | None = None,
        tools_max_rounds: int = 3,
    ):
        super().__init__()
        self.model = model
        self._api_key = api_key
        self._tools_max_rounds = max(tools_max_rounds, 1)
        self._pending_tool_calls: dict[int, dict[str, str]] = {}

    async def simple_response(
        self,
        text: str,
        participant: object | None = None,
    ) -> LLMResponseEvent:
        if self._conversation is None:
            logger.warning(
                'Cannot request a response from "%s" '
                "- conversation not initialized yet.",
                self.model,
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
        messages: list[dict[str, object]] | None = None,
        stream: bool = True,
        **kwargs: object,
    ) -> LLMResponseEvent:
        if messages is None:
            messages = await self._build_model_request()

        tools_param = None
        tools_spec = self.get_available_functions()
        if tools_spec:
            tools_param = self._convert_tools_to_provider_format(tools_spec)

        return await self._create_response_internal(
            messages=messages,
            tools=tools_param,
            stream=stream,
            **kwargs,
        )

    async def _create_response_internal(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        stream: bool = True,
        **kwargs: object,
    ) -> LLMResponseEvent:
        model = str(kwargs.get("model", self.model))

        params: dict[str, object] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "drop_params": True,
        }
        if self._api_key:
            params["api_key"] = self._api_key
        if tools:
            params["tools"] = tools

        self.events.send(
            LLMRequestStartedEvent(
                plugin_name=PLUGIN_NAME,
                model=model,
                streaming=stream,
            )
        )

        request_start = time.perf_counter()

        try:
            if stream:
                return await self._handle_streaming(params, request_start)
            return await self._handle_non_streaming(params, request_start)
        except litellm.exceptions.AuthenticationError as exc:
            logger.exception("LiteLLM auth error for model %s", model)
            return LLMResponseEvent(original=None, text="", exception=exc)
        except litellm.exceptions.RateLimitError as exc:
            logger.exception("LiteLLM rate limit for model %s", model)
            return LLMResponseEvent(original=None, text="", exception=exc)
        except litellm.exceptions.APIConnectionError as exc:
            logger.exception("LiteLLM connection error for model %s", model)
            return LLMResponseEvent(original=None, text="", exception=exc)
        except litellm.exceptions.APIError as exc:
            logger.exception("LiteLLM API error for model %s", model)
            return LLMResponseEvent(original=None, text="", exception=exc)

    async def _handle_streaming(
        self, params: dict[str, object], request_start: float
    ) -> LLMResponseEvent:
        response = await litellm.acompletion(**params)

        content_parts: list[str] = []
        first_token_time: float | None = None
        self._pending_tool_calls = {}

        async for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if delta.content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                content_parts.append(delta.content)
                self.events.send(
                    LLMResponseChunkEvent(
                        plugin_name=PLUGIN_NAME,
                        text=delta.content,
                    )
                )

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = getattr(tc, "index", 0)
                    entry = self._pending_tool_calls.setdefault(
                        idx, {"id": "", "name": "", "arguments": ""}
                    )
                    if tc.id:
                        entry["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            entry["name"] = tc.function.name
                        if tc.function.arguments:
                            entry["arguments"] += tc.function.arguments

        full_text = "".join(content_parts)
        total_time = time.perf_counter() - request_start
        ttft = (first_token_time - request_start) if first_token_time else total_time

        self.events.send(
            LLMResponseCompletedEvent(
                plugin_name=PLUGIN_NAME,
                text=full_text,
                ttft=ttft,
                duration=total_time,
            )
        )

        return LLMResponseEvent(original=None, text=full_text)

    async def _handle_non_streaming(
        self, params: dict[str, object], request_start: float
    ) -> LLMResponseEvent:
        response = await litellm.acompletion(**params)
        content = response.choices[0].message.content or ""
        total_time = time.perf_counter() - request_start

        self.events.send(
            LLMResponseCompletedEvent(
                plugin_name=PLUGIN_NAME,
                text=content,
                ttft=total_time,
                duration=total_time,
            )
        )

        return LLMResponseEvent(original=response, text=content)

    async def _build_model_request(self) -> list[dict[str, object]]:
        messages: list[dict[str, object]] = []
        if self._instructions:
            messages.append({"role": "system", "content": self._instructions})
        if self._conversation:
            messages.extend(await self._conversation.get_messages())
        return messages

    def _convert_tools_to_provider_format(
        self, tools: list[ToolSchema]
    ) -> list[dict[str, object]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    def _extract_tool_calls_from_response(
        self, response: object
    ) -> list[NormalizedToolCallItem]:
        choices = getattr(response, "choices", None)
        if not choices:
            return []
        message = choices[0].message
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            return []
        return [
            NormalizedToolCallItem(
                id=tc.id,
                name=tc.function.name,
                arguments_json=json.loads(tc.function.arguments),
            )
            for tc in tool_calls
        ]
