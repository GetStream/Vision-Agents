"""Shared utilities and base classes for local-inference LLM plugins.

Provides:
- Tool-call parsing from raw model text output.
- Message building from instructions + conversation.
- ``LocalTextLLM`` — abstract base for text-only local LLMs (Transformers, MLX).
"""

import abc
import asyncio
import json
import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, Generic, Optional, TypeVar

from vision_agents.core.agents.conversation import Conversation
from vision_agents.core.llm.events import (
    LLMRequestStartedEvent,
    LLMResponseChunkEvent,
    LLMResponseCompletedEvent,
)
from vision_agents.core.llm.llm import LLM, LLMResponseEvent
from vision_agents.core.llm.llm_types import NormalizedToolCallItem, ToolSchema
from vision_agents.core.warmup import Warmable

from . import events
from ._tool_call_loop import (
    convert_tools_to_chat_completions_format,
    run_tool_call_loop,
)

logger = logging.getLogger(__name__)

R = TypeVar("R")


def _extract_json_objects(text: str) -> list[dict[str, Any]]:
    """Extract all top-level JSON objects from *text* using ``raw_decode``."""
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    idx = 0
    while idx < len(text):
        idx = text.find("{", idx)
        if idx == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
            if isinstance(obj, dict):
                objects.append(obj)
            idx = end
        except json.JSONDecodeError:
            idx += 1
    return objects


def extract_tool_calls_from_text(text: str) -> list[NormalizedToolCallItem]:
    """Parse tool calls from raw model output text.

    Supports:
    - Hermes format: ``<tool_call>{"name": ..., "arguments": ...}</tool_call>``
    - Generic JSON: ``{"name": ..., "arguments": ...}``
    """
    tool_calls: list[NormalizedToolCallItem] = []

    hermes_pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    for match in re.finditer(hermes_pattern, text, re.DOTALL):
        for obj in _extract_json_objects(match.group(1)):
            tool_calls.append(
                {
                    "type": "tool_call",
                    "id": obj.get("id", str(uuid.uuid4())),
                    "name": obj.get("name", ""),
                    "arguments_json": obj.get("arguments", {}),
                }
            )

    if tool_calls:
        return tool_calls

    for obj in _extract_json_objects(text):
        if "name" in obj and "arguments" in obj:
            tool_calls.append(
                {
                    "type": "tool_call",
                    "id": str(uuid.uuid4()),
                    "name": obj["name"],
                    "arguments_json": obj["arguments"],
                }
            )

    return tool_calls


def build_messages(
    instructions: str,
    conversation: Optional[Conversation],
) -> list[dict[str, Any]]:
    """Build a chat message list from instructions and conversation history."""
    messages: list[dict[str, Any]] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    if conversation:
        for msg in conversation.messages:
            messages.append({"role": msg.role, "content": msg.content})
    return messages


async def run_local_tool_call_loop(
    llm: LLM,
    tool_calls: list[NormalizedToolCallItem],
    messages: list[dict[str, Any]],
    create_followup: Callable[[list[dict[str, Any]]], Awaitable[LLMResponseEvent]],
    *,
    max_tool_rounds: int = 3,
) -> LLMResponseEvent:
    """Run a multi-round tool-call loop with text-based tool-call extraction.

    Wraps ``run_tool_call_loop`` with the text-parsing step that all local
    inference backends share: after each follow-up generation, tool calls
    are extracted from the raw model output via ``extract_tool_calls_from_text``.
    """

    async def _generate_followup(
        current_messages: list[dict[str, Any]],
    ) -> tuple[LLMResponseEvent, list[NormalizedToolCallItem]]:
        result = await create_followup(current_messages)
        next_calls = extract_tool_calls_from_text(result.text)
        return result, next_calls

    return await run_tool_call_loop(
        llm,
        tool_calls,
        messages,
        _generate_followup,
        max_rounds=max_tool_rounds,
    )


class LocalTextLLM(LLM, Warmable[R], Generic[R]):
    """Abstract base for local text LLM inference (Transformers, MLX, etc.).

    Subclasses implement model loading, chat template application, and the
    actual generate calls. This base provides the shared orchestration:
    warmup, ``simple_response``, ``create_response`` (template method with
    tool-call suppression), tool handling, and lifecycle management.
    """

    _plugin_name: str

    def __init__(
        self,
        model: str,
        max_new_tokens: int = 512,
        max_tool_rounds: int = 3,
    ):
        super().__init__()
        self.model_id = model
        self._max_new_tokens = max_new_tokens
        self._max_tool_rounds = max_tool_rounds
        self._resources: Optional[R] = None
        self.events.register_events_from_module(events)

    @abc.abstractmethod
    def _load_model_sync(self) -> R:
        """Load and return model resources (called in a thread)."""

    @abc.abstractmethod
    def _apply_template(
        self,
        messages: list[dict[str, Any]],
        tools_param: Optional[list[dict[str, Any]]],
    ) -> tuple[Any, bool] | None:
        """Apply the chat template to produce model input.

        Returns ``(prepared_input, tools_applied)`` on success, where
        *tools_applied* indicates whether the template accepted the tools
        parameter. Returns ``None`` on unrecoverable failure.
        """

    @abc.abstractmethod
    async def _generate_streaming(
        self, prepared_input: Any, max_tokens: int, temperature: float
    ) -> LLMResponseEvent:
        """Generate a streaming response from *prepared_input*."""

    @abc.abstractmethod
    async def _generate_non_streaming(
        self,
        prepared_input: Any,
        max_tokens: int,
        temperature: float,
        emit_events: bool,
    ) -> LLMResponseEvent:
        """Generate a non-streaming response from *prepared_input*."""

    async def on_warmup(self) -> R:
        logger.info("Loading model: %s", self.model_id)
        resources = await asyncio.to_thread(self._load_model_sync)
        logger.info("Model loaded: %s", self.model_id)
        return resources

    def on_warmed_up(self, resource: R) -> None:
        self._resources = resource

    async def simple_response(
        self,
        text: str,
        participant: Optional[Any] = None,
    ) -> LLMResponseEvent:
        if self._conversation is None:
            logger.warning(
                "Conversation not initialized. Call set_conversation() first."
            )
            return LLMResponseEvent(original=None, text="")

        if participant is None:
            await self._conversation.send_message(
                role="user", user_id="user", content=text
            )

        messages = self._build_messages()
        return await self.create_response(messages=messages, stream=True)

    async def create_response(
        self,
        messages: Optional[list[dict[str, Any]]] = None,
        *,
        stream: bool = True,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponseEvent:
        if self._resources is None:
            logger.error("Model not loaded. Ensure warmup() was called.")
            return LLMResponseEvent(original=None, text="")

        if messages is None:
            messages = self._build_messages()

        tools_param: Optional[list[dict[str, Any]]] = None
        tools_spec = self.get_available_functions()
        if tools_spec:
            tools_param = convert_tools_to_chat_completions_format(tools_spec)

        template_result = self._apply_template(messages, tools_param)
        if template_result is None:
            return LLMResponseEvent(original=None, text="")
        prepared_input, tools_applied = template_result

        max_tokens = max_new_tokens or self._max_new_tokens
        is_tool_followup = kwargs.pop("_tool_followup", False)
        suppress_events = tools_applied

        self.events.send(
            LLMRequestStartedEvent(
                plugin_name=self._plugin_name,
                model=self.model_id,
                streaming=stream and not suppress_events,
            )
        )

        if stream and not suppress_events:
            result = await self._generate_streaming(
                prepared_input, max_tokens, temperature
            )
        else:
            result = await self._generate_non_streaming(
                prepared_input, max_tokens, temperature, not suppress_events
            )

        if suppress_events and result.text:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Raw model output (tools registered): %s", result.text)
            tool_calls = extract_tool_calls_from_text(result.text)
            if tool_calls:
                if is_tool_followup:
                    return result
                return await self._handle_tool_calls(tool_calls, messages, kwargs)
            response_id = str(uuid.uuid4())
            if stream:
                self.events.send(
                    LLMResponseChunkEvent(
                        plugin_name=self._plugin_name,
                        content_index=None,
                        item_id=response_id,
                        output_index=0,
                        sequence_number=0,
                        delta=result.text,
                        is_first_chunk=True,
                        time_to_first_token_ms=None,
                    )
                )
            self.events.send(
                LLMResponseCompletedEvent(
                    plugin_name=self._plugin_name,
                    original=None,
                    text=result.text,
                    item_id=response_id,
                    model=self.model_id,
                )
            )

        return result

    def _build_messages(self) -> list[dict[str, Any]]:
        return build_messages(self._instructions, self._conversation)

    def _convert_tools_to_provider_format(
        self, tools: list[ToolSchema]
    ) -> list[dict[str, Any]]:
        return convert_tools_to_chat_completions_format(tools)

    async def _handle_tool_calls(
        self,
        tool_calls: list[NormalizedToolCallItem],
        messages: list[dict[str, Any]],
        kwargs: dict[str, Any],
    ) -> LLMResponseEvent:
        async def _followup(msgs: list[dict[str, Any]]) -> LLMResponseEvent:
            return await self.create_response(
                messages=msgs, _tool_followup=True, **kwargs
            )

        return await run_local_tool_call_loop(
            self, tool_calls, messages, _followup, max_tool_rounds=self._max_tool_rounds
        )

    def unload(self) -> None:
        logger.info("Unloading model: %s", self.model_id)
        self._resources = None

    @property
    def is_loaded(self) -> bool:
        return self._resources is not None
