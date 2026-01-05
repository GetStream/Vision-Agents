import json
import logging
from typing import Any, Dict, List, Optional

from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant
from huggingface_hub import AsyncInferenceClient
from huggingface_hub.inference._providers import PROVIDER_OR_POLICY_T
from vision_agents.core.llm.events import (
    LLMResponseChunkEvent,
    LLMResponseCompletedEvent,
)
from vision_agents.core.llm.llm import LLM, LLMResponseEvent
from vision_agents.core.llm.llm_types import NormalizedToolCallItem, ToolSchema
from vision_agents.core.processors import Processor

from . import events

logger = logging.getLogger(__name__)


PLUGIN_NAME = "huggingface_llm"


class HuggingFaceLLM(LLM):
    """
    HuggingFace Inference integration for text-only LLM models.

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
        client: Optional[AsyncInferenceClient] = None,
    ):
        """
        Initialize the HuggingFaceLLM class.

        Args:
            model: The HuggingFace model ID to use.
            api_key: HuggingFace API token. Defaults to HF_TOKEN environment variable.
            provider: Inference provider (e.g., "together", "groq", "fireworks-ai"). Defaults to "auto" which auto-selects.
            client: Optional AsyncInferenceClient instance for dependency injection.
        """
        super().__init__()
        self.model = model
        self.provider = provider
        self.events.register_events_from_module(events)
        self._pending_tool_calls: Dict[int, Dict[str, Any]] = {}

        if client is not None:
            self._client = client
        else:
            self._client = AsyncInferenceClient(
                token=api_key,
                model=model,
                provider=provider,
            )

    async def simple_response(
        self,
        text: str,
        processors: Optional[list[Processor]] = None,
        participant: Optional[Participant] = None,
    ) -> LLMResponseEvent:
        """
        Create an LLM response from text input.

        This method is called when a new STT transcript is received.

        Args:
            text: The text to respond to.
            processors: List of processors with video/voice AI state.
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
        """
        Create a response using HuggingFace's Inference API.

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
            tools_param = self._convert_tools_to_provider_format(tools_spec)

        return await self._create_response_internal(
            messages=messages,
            tools=tools_param,
            stream=stream,
            **kwargs,
        )

    async def _create_response_internal(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = True,
        **kwargs: Any,
    ) -> LLMResponseEvent:
        """Internal method to create response with tool handling loop."""
        request_kwargs: Dict[str, Any] = {
            "messages": messages,
            "model": kwargs.get("model", self.model),
            "stream": stream,
        }
        if tools:
            request_kwargs["tools"] = tools

        try:
            response = await self._client.chat.completions.create(**request_kwargs)
        except Exception as e:
            logger.exception(f'Failed to get a response from the LLM "{self.model}"')
            self.events.send(
                events.LLMErrorEvent(
                    plugin_name=PLUGIN_NAME,
                    error_message=str(e),
                    event_data=e,
                )
            )
            return LLMResponseEvent(original=None, text="")

        if stream:
            return await self._process_streaming_response(
                response, messages, tools, kwargs
            )
        else:
            return await self._process_non_streaming_response(
                response, messages, tools, kwargs
            )

    async def _process_streaming_response(
        self,
        response: Any,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        kwargs: Dict[str, Any],
    ) -> LLMResponseEvent:
        """Process a streaming response, handling tool calls if present."""
        llm_response: LLMResponseEvent = LLMResponseEvent(original=None, text="")
        text_chunks: list[str] = []
        total_text = ""
        self._pending_tool_calls = {}
        accumulated_tool_calls: List[NormalizedToolCallItem] = []
        i = 0
        chunk_id = ""

        async for chunk in response:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            content = choice.delta.content if choice.delta else None
            finish_reason = choice.finish_reason
            chunk_id = chunk.id if chunk.id else chunk_id

            if choice.delta and choice.delta.tool_calls:
                for tc in choice.delta.tool_calls:
                    self._accumulate_tool_call_chunk(tc)

            if content:
                text_chunks.append(content)
                self.events.send(
                    LLMResponseChunkEvent(
                        plugin_name=PLUGIN_NAME,
                        content_index=None,
                        item_id=chunk_id,
                        output_index=0,
                        sequence_number=i,
                        delta=content,
                    )
                )

            if finish_reason:
                if finish_reason in ("length", "content"):
                    logger.warning(
                        f'The model finished the response due to reason "{finish_reason}"'
                    )

                if finish_reason == "tool_calls":
                    accumulated_tool_calls = self._finalize_pending_tool_calls()

                total_text = "".join(text_chunks)
                self.events.send(
                    LLMResponseCompletedEvent(
                        plugin_name=PLUGIN_NAME,
                        original=chunk,
                        text=total_text,
                        item_id=chunk_id,
                    )
                )

            llm_response = LLMResponseEvent(original=chunk, text=total_text)
            i += 1

        if accumulated_tool_calls:
            return await self._handle_tool_calls(
                accumulated_tool_calls, messages, tools, kwargs
            )

        return llm_response

    async def _process_non_streaming_response(
        self,
        response: Any,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        kwargs: Dict[str, Any],
    ) -> LLMResponseEvent:
        """Process a non-streaming response, handling tool calls if present."""
        text = response.choices[0].message.content or ""
        llm_response = LLMResponseEvent(original=response, text=text)

        tool_calls = self._extract_tool_calls_from_response(response)
        if tool_calls:
            return await self._handle_tool_calls(tool_calls, messages, tools, kwargs)

        self.events.send(
            LLMResponseCompletedEvent(
                plugin_name=PLUGIN_NAME,
                original=response,
                text=text,
                item_id=response.id,
            )
        )
        return llm_response

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

    def _accumulate_tool_call_chunk(self, tc_chunk: Any) -> None:
        """Accumulate tool call data from streaming chunks."""
        idx = tc_chunk.index
        if idx not in self._pending_tool_calls:
            self._pending_tool_calls[idx] = {
                "id": tc_chunk.id or "",
                "name": "",
                "arguments_parts": [],
            }

        pending = self._pending_tool_calls[idx]
        if tc_chunk.id:
            pending["id"] = tc_chunk.id
        if tc_chunk.function:
            if tc_chunk.function.name:
                pending["name"] = tc_chunk.function.name
            if tc_chunk.function.arguments:
                pending["arguments_parts"].append(tc_chunk.function.arguments)

    def _finalize_pending_tool_calls(self) -> List[NormalizedToolCallItem]:
        """Convert accumulated tool call chunks into normalized tool calls."""
        tool_calls: List[NormalizedToolCallItem] = []
        for pending in self._pending_tool_calls.values():
            args_str = "".join(pending["arguments_parts"]).strip() or "{}"
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {}

            tool_call: NormalizedToolCallItem = {
                "type": "tool_call",
                "id": pending["id"],
                "name": pending["name"],
                "arguments_json": args,
            }
            tool_calls.append(tool_call)

        self._pending_tool_calls = {}
        return tool_calls

    def _convert_tools_to_provider_format(
        self, tools: List[ToolSchema]
    ) -> List[Dict[str, Any]]:
        """Convert ToolSchema objects to Chat Completions API format."""
        result = []
        for t in tools or []:
            name = t.get("name", "unnamed_tool")
            description = t.get("description", "") or ""
            params = t.get("parameters_schema") or t.get("parameters") or {}
            if not isinstance(params, dict):
                params = {}
            params.setdefault("type", "object")
            params.setdefault("properties", {})

            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": params,
                    },
                }
            )
        return result

    def _extract_tool_calls_from_response(
        self, response: Any
    ) -> List[NormalizedToolCallItem]:
        """Extract tool calls from a non-streaming response."""
        tool_calls: List[NormalizedToolCallItem] = []

        if not response.choices:
            return tool_calls

        message = response.choices[0].message
        if not message.tool_calls:
            return tool_calls

        for tc in message.tool_calls:
            args_str = tc.function.arguments or "{}"
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {}

            tool_call: NormalizedToolCallItem = {
                "type": "tool_call",
                "id": tc.id,
                "name": tc.function.name,
                "arguments_json": args,
            }
            tool_calls.append(tool_call)

        return tool_calls

    async def _handle_tool_calls(
        self,
        tool_calls: List[NormalizedToolCallItem],
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        kwargs: Dict[str, Any],
    ) -> LLMResponseEvent:
        """Execute tool calls and get follow-up response."""
        llm_response: LLMResponseEvent = LLMResponseEvent(original=None, text="")
        max_rounds = 3
        current_tool_calls = tool_calls
        seen: set[tuple] = set()
        current_messages = list(messages)

        for round_num in range(max_rounds):
            triples, seen = await self._dedup_and_execute(
                current_tool_calls,
                max_concurrency=8,
                timeout_s=30,
                seen=seen,
            )

            if not triples:
                break

            assistant_tool_calls = []
            tool_results = []
            for tc, res, err in triples:
                cid = tc.get("id")
                if not cid:
                    continue

                assistant_tool_calls.append(
                    {
                        "id": cid,
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc.get("arguments_json", {})),
                        },
                    }
                )
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": cid,
                        "content": self._sanitize_tool_output(
                            err if err is not None else res
                        ),
                    }
                )

            if not tool_results:
                return llm_response

            current_messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": assistant_tool_calls,
                }
            )
            current_messages.extend(tool_results)

            request_kwargs: Dict[str, Any] = {
                "messages": current_messages,
                "model": kwargs.get("model", self.model),
                "stream": True,
            }
            if tools:
                request_kwargs["tools"] = tools

            try:
                follow_up = await self._client.chat.completions.create(**request_kwargs)
            except Exception as e:
                logger.exception("Failed to get follow-up response")
                self.events.send(
                    events.LLMErrorEvent(
                        plugin_name=PLUGIN_NAME,
                        error_message=str(e),
                        event_data=e,
                    )
                )
                return llm_response

            text_chunks: list[str] = []
            self._pending_tool_calls = {}
            next_tool_calls: List[NormalizedToolCallItem] = []
            i = 0
            chunk_id = ""

            async for chunk in follow_up:
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                content = choice.delta.content if choice.delta else None
                finish_reason = choice.finish_reason
                chunk_id = chunk.id if chunk.id else chunk_id

                if choice.delta and choice.delta.tool_calls:
                    for tc in choice.delta.tool_calls:
                        self._accumulate_tool_call_chunk(tc)

                if content:
                    text_chunks.append(content)
                    self.events.send(
                        LLMResponseChunkEvent(
                            plugin_name=PLUGIN_NAME,
                            content_index=None,
                            item_id=chunk_id,
                            output_index=0,
                            sequence_number=i,
                            delta=content,
                        )
                    )

                if finish_reason:
                    if finish_reason == "tool_calls":
                        next_tool_calls = self._finalize_pending_tool_calls()

                    total_text = "".join(text_chunks)
                    self.events.send(
                        LLMResponseCompletedEvent(
                            plugin_name=PLUGIN_NAME,
                            original=chunk,
                            text=total_text,
                            item_id=chunk_id,
                        )
                    )
                    llm_response = LLMResponseEvent(original=chunk, text=total_text)

                i += 1

            if next_tool_calls and round_num < max_rounds - 1:
                current_tool_calls = next_tool_calls
                continue

            return llm_response

        return llm_response
