import json
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from vision_agents.core.agents.conversation import Message
from vision_agents.core.edge.types import Participant
from vision_agents.core.llm.llm import LLM, LLMResponseDelta, LLMResponseFinal
from vision_agents.core.llm.llm_types import NormalizedToolCallItem, ToolSchema

import anthropic
from anthropic import AsyncAnthropic, AsyncStream
from anthropic.types import (
    Message as ClaudeMessage,
)
from anthropic.types import (
    RawContentBlockDeltaEvent,
    RawMessageStreamEvent,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"


class ClaudeLLM(LLM):
    """
    The ClaudeLLM class provides full/native access to the claude SDK methods.
    It only standardized the minimal feature set that's needed for the agent integration.

    The agent requires that we standardize:
    - sharing instructions
    - keeping conversation history
    - response normalization

    Notes on the Claude integration
    - the native method is called create_message (maps 1-1 to messages.create)
    - history is maintained manually by keeping it in memory

    Examples:

        from vision_agents.plugins import anthropic
        llm = anthropic.LLM(model="claude-opus-4-1-20250805")
    """

    provider_name = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        client: Optional[AsyncAnthropic] = None,
        tools_max_rounds: int = 3,
    ):
        """
        Initialize the ClaudeLLM class.

        Args:
            model (str): The model to use. https://docs.anthropic.com/en/docs/about-claude/models/overview.
                Default - `"claude-sonnet-4-6"`.
            api_key: optional API key. by default loads from ANTHROPIC_API_KEY
            client: optional Anthropic client. by default creates a new client object.
            tools_max_rounds: max calling rounds for multi-hop tool call. Default - ``3``.
        """
        super().__init__()
        self.model = model
        self._tools_max_rounds = max(tools_max_rounds, 1)
        self._pending_tool_uses_by_index: Dict[
            int, Dict[str, Any]
        ] = {}  # index -> {id, name, parts: []}
        self.client = client or anthropic.AsyncAnthropic(api_key=api_key)

    async def simple_response(
        self,
        text: str,
        participant: Optional[Participant] = None,
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        """
        simple_response is a standardized way (across openai, claude, gemini etc.) to create a response.

        Args:
            text: The text to respond to
            participant: optionally the participant object

        Examples:

            llm.simple_response("say hi to the user, be mean")
        """
        async for item in self._stream_message(
            messages=[{"role": "user", "content": text}], max_tokens=1000
        ):
            yield item

    async def create_message(self, *args, **kwargs) -> LLMResponseFinal:
        """
        create_message gives you full support/access to the native Claude messages.create method.

        This wrapper drains the internal response stream and returns the final
        text plus the original Claude response object for compatibility with the
        native-style API.
        """
        final_response: Optional[LLMResponseFinal] = None
        async for item in self._stream_message(*args, **kwargs):
            if isinstance(item, LLMResponseFinal):
                final_response = item

        if final_response is None:
            raise RuntimeError("Claude stream ended without a final response")

        return final_response

    async def _stream_message(
        self, *args, **kwargs
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        """
        Internal streaming implementation shared by simple_response and create_message.
        """
        if "model" not in kwargs:
            kwargs["model"] = self.model

        if "stream" not in kwargs:
            kwargs["stream"] = True

        if self._instructions and "system" not in kwargs:
            kwargs["system"] = self._instructions

        # Add tools if available - use Anthropic format
        tools = self.get_available_functions()
        if tools:
            kwargs["tools"] = self._convert_tools_to_provider_format(tools)
            kwargs.setdefault("tool_choice", {"type": "auto"})

        if "messages" not in kwargs:
            raise ValueError("messages are required")
        kwargs["messages"] = self._apply_conversation_history(kwargs["messages"])

        # Note: Message history is tracked in _conversation, no need to emit as event here

        # Track timing
        request_start_time = time.perf_counter()
        first_token_time: Optional[float] = None

        self._pending_tool_uses_by_index.clear()
        try:
            original = await self.client.messages.create(*args, **kwargs)
        except anthropic.APIError as e:
            logger.exception(f'Failed to get a response from the LLM "{self.model}"')
            self.on_llm_error(error=e)
            yield LLMResponseFinal(original=None, text="")
            return
        if isinstance(original, ClaudeMessage):
            # Extract text from Claude's response format - safely handle all text blocks
            final_original = original
            final_text = self._concat_text_blocks(original.content)

            # Multi-hop tool calling loop for non-streaming
            function_calls = self._extract_tool_calls_from_response(original)
            if function_calls:
                messages = kwargs["messages"][:]
                rounds = 0
                seen: set[tuple[str, str, str]] = set()
                current_calls = function_calls

                while current_calls and rounds < self._tools_max_rounds:
                    # Execute calls concurrently with dedup
                    triples, seen = await self._dedup_and_execute(
                        current_calls, seen=seen, max_concurrency=8, timeout_s=30
                    )  # type: ignore[arg-type]

                    if not triples:
                        break

                    # Build tool_result user message
                    assistant_content = []
                    tool_result_blocks = []
                    for tc, res, err in triples:
                        assistant_content.append(
                            {
                                "type": "tool_use",
                                "id": tc["id"],
                                "name": tc["name"],
                                "input": tc["arguments_json"],
                            }
                        )

                        payload = self._sanitize_tool_output(res)
                        tool_result_blocks.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tc["id"],
                                "content": payload,
                            }
                        )

                    assistant_msg = {"role": "assistant", "content": assistant_content}
                    user_tool_results_msg = {
                        "role": "user",
                        "content": tool_result_blocks,
                    }
                    messages = messages + [assistant_msg, user_tool_results_msg]

                    # Ask again WITH tools so Claude can do another hop
                    tools_cfg = {
                        "tools": self._convert_tools_to_provider_format(
                            self.get_available_functions()
                        ),
                        "tool_choice": {"type": "auto"},
                        "stream": False,
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": 1000,
                    }

                    follow_up_response = await self.client.messages.create(**tools_cfg)

                    # Extract new tool calls from follow-up response
                    current_calls = self._extract_tool_calls_from_response(
                        follow_up_response
                    )
                    final_original = follow_up_response
                    final_text = self._concat_text_blocks(follow_up_response.content)
                    rounds += 1

                # Finalization pass: no tools so Claude must answer in text
                if current_calls or rounds > 0:  # Only if we had tool calls
                    final_response = await self.client.messages.create(
                        model=self.model,
                        messages=messages,  # includes assistant tool_use + user tool_result blocks
                        stream=False,
                        max_tokens=1000,
                    )
                    final_original = final_response
                    final_text = self._concat_text_blocks(final_response.content)

            latency_ms = (time.perf_counter() - request_start_time) * 1000
            usage = getattr(final_original, "usage", None)
            input_tokens = getattr(usage, "input_tokens", None)
            output_tokens = getattr(usage, "output_tokens", None)
            yield LLMResponseFinal(
                original=final_original,
                text=final_text,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=(input_tokens or 0) + (output_tokens or 0)
                if input_tokens or output_tokens
                else None,
                model=getattr(final_original, "model", None) or self.model,
            )

        elif isinstance(original, AsyncStream):
            stream: AsyncStream[RawMessageStreamEvent] = original
            text_parts: List[str] = []
            accumulated_calls: List[NormalizedToolCallItem] = []
            # Track if we've emitted the first chunk for the entire request
            emitted_first_chunk = False
            # Track usage from message_start and message_delta events
            input_tokens = None
            output_tokens = None
            sequence_number = 0

            # 1) First round: read stream, gather initial tool_use calls
            async for event in stream:
                # Track usage from streaming events
                if (
                    event.type == "message_start"
                    and event.message
                    and event.message.usage
                ):
                    input_tokens = event.message.usage.input_tokens
                elif event.type == "message_delta" and event.usage:
                    output_tokens = event.usage.output_tokens

                text_delta = self._extract_text_delta(event)
                if text_delta is not None:
                    content_index, delta_text = text_delta
                    text_parts.append(delta_text)
                    if first_token_time is None:
                        first_token_time = time.perf_counter()

                    is_first_chunk = not emitted_first_chunk
                    ttft_ms = (
                        (first_token_time - request_start_time) * 1000
                        if is_first_chunk
                        else None
                    )
                    yield self._build_response_delta(
                        content_index=content_index,
                        delta=delta_text,
                        sequence_number=sequence_number,
                        is_first_chunk=is_first_chunk,
                        time_to_first_token_ms=ttft_ms,
                    )
                    emitted_first_chunk = True
                    sequence_number += 1
                # Collect tool_use calls as they complete (your helper already reconstructs args)
                new_calls, _ = self._extract_tool_calls_from_stream_chunk(event, None)
                if new_calls:
                    accumulated_calls.extend(new_calls)

            # Track full message history to reuse across rounds
            messages = kwargs["messages"][:]  # start from prior history
            rounds = 0
            seen = set()

            # 2) While there are tool calls, execute -> return tool_result -> ask again (with tools)
            last_followup_stream = None
            while accumulated_calls and rounds < self._tools_max_rounds:
                # Execute calls concurrently with dedup
                triples, seen = await self._dedup_and_execute(
                    accumulated_calls, seen=seen, max_concurrency=8, timeout_s=30
                )  # type: ignore[arg-type]

                # Build tool_result user message
                # Also reconstruct the assistant tool_use message that triggered these calls
                assistant_content = []
                for tc, res, err in triples:
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["name"],
                            "input": tc["arguments_json"],
                        }
                    )

                # tool_result blocks (sanitize to keep payloads safe)
                tool_result_blocks = []
                for tc, res, err in triples:
                    payload = self._sanitize_tool_output(res)
                    tool_result_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tc["id"],
                            "content": payload,
                        }
                    )

                assistant_msg = {"role": "assistant", "content": assistant_content}
                user_tool_results_msg = {"role": "user", "content": tool_result_blocks}
                messages = messages + [assistant_msg, user_tool_results_msg]

                # Ask again WITH tools so Claude can do another hop
                tools_cfg = {
                    "tools": self._convert_tools_to_provider_format(
                        self.get_available_functions()
                    ),
                    "tool_choice": {"type": "auto"},
                    "stream": True,
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 1000,
                }

                follow_up_stream = await self.client.messages.create(**tools_cfg)

                # Read the follow-up stream; collect text deltas & any NEW tool_use calls
                follow_up_text_parts: List[str] = []
                accumulated_calls = []  # reset; we'll refill with new calls
                async for ev in follow_up_stream:
                    last_followup_stream = ev
                    # Track usage from streaming events
                    if ev.type == "message_start" and ev.message and ev.message.usage:
                        input_tokens = ev.message.usage.input_tokens
                    elif ev.type == "message_delta" and ev.usage:
                        output_tokens = ev.usage.output_tokens

                    text_delta = self._extract_text_delta(ev)
                    if text_delta is not None:
                        content_index, delta_text = text_delta
                        follow_up_text_parts.append(delta_text)
                        if first_token_time is None:
                            first_token_time = time.perf_counter()

                        is_first_chunk = not emitted_first_chunk
                        ttft_ms = (
                            (first_token_time - request_start_time) * 1000
                            if is_first_chunk
                            else None
                        )
                        yield self._build_response_delta(
                            content_index=content_index,
                            delta=delta_text,
                            sequence_number=sequence_number,
                            is_first_chunk=is_first_chunk,
                            time_to_first_token_ms=ttft_ms,
                        )
                        emitted_first_chunk = True
                        sequence_number += 1
                    new_calls, _ = self._extract_tool_calls_from_stream_chunk(ev, None)
                    if new_calls:
                        accumulated_calls.extend(new_calls)

                # append emergent text so far
                if follow_up_text_parts:
                    text_parts.append("".join(follow_up_text_parts))

                rounds += 1

            # 3) Finalization pass: no tools so Claude must answer in text
            if accumulated_calls or rounds > 0:  # Only if we had tool calls
                final_stream = await self.client.messages.create(
                    model=self.model,
                    messages=messages,  # includes assistant tool_use + user tool_result blocks
                    stream=True,
                    max_tokens=1000,
                )
                final_text_parts: List[str] = []
                async for ev in final_stream:
                    last_followup_stream = ev
                    # Track usage from streaming events
                    if ev.type == "message_start" and ev.message and ev.message.usage:
                        input_tokens = ev.message.usage.input_tokens
                    elif ev.type == "message_delta" and ev.usage:
                        output_tokens = ev.usage.output_tokens

                    text_delta = self._extract_text_delta(ev)
                    if text_delta is not None:
                        content_index, delta_text = text_delta
                        final_text_parts.append(delta_text)
                        if first_token_time is None:
                            first_token_time = time.perf_counter()

                        is_first_chunk = not emitted_first_chunk
                        ttft_ms = (
                            (first_token_time - request_start_time) * 1000
                            if is_first_chunk
                            else None
                        )
                        yield self._build_response_delta(
                            content_index=content_index,
                            delta=delta_text,
                            sequence_number=sequence_number,
                            is_first_chunk=is_first_chunk,
                            time_to_first_token_ms=ttft_ms,
                        )
                        emitted_first_chunk = True
                        sequence_number += 1
                if final_text_parts:
                    text_parts.append("".join(final_text_parts))

            # 4) Done -> yield all collected text
            total_text = "".join(text_parts)

            # Calculate timing metrics
            latency_ms = (time.perf_counter() - request_start_time) * 1000
            ttft_ms = None
            if first_token_time is not None:
                ttft_ms = (first_token_time - request_start_time) * 1000

            yield LLMResponseFinal(
                original=last_followup_stream or original,
                text=total_text,
                latency_ms=latency_ms,
                time_to_first_token_ms=ttft_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=(input_tokens or 0) + (output_tokens or 0)
                if input_tokens or output_tokens
                else None,
                model=self.model,
            )

    async def close(self) -> None:
        await self.client.close()

    def _extract_text_delta(
        self, event: RawMessageStreamEvent
    ) -> tuple[int, str] | None:
        if event.type != "content_block_delta":
            return None

        delta_event: RawContentBlockDeltaEvent = event
        if hasattr(delta_event.delta, "text") and delta_event.delta.text:
            return delta_event.index, delta_event.delta.text
        return None

    def _build_response_delta(
        self,
        *,
        content_index: int,
        delta: str,
        sequence_number: int,
        is_first_chunk: bool,
        time_to_first_token_ms: Optional[float],
    ) -> LLMResponseDelta:
        return LLMResponseDelta(
            content_index=content_index,
            output_index=0,
            sequence_number=sequence_number,
            delta=delta,
            is_first_chunk=is_first_chunk,
            time_to_first_token_ms=time_to_first_token_ms,
        )

    @staticmethod
    def _normalize_message(claude_messages: Any) -> List["Message"]:
        return normalize_claude_messages(claude_messages)

    def _apply_conversation_history(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not self._conversation:
            return messages

        old_messages = [
            m.original
            if isinstance(m.original, dict)
            else {"role": m.role or "user", "content": m.content or ""}
            for m in self._conversation.messages
        ]
        effective_messages = self._merge_messages(old_messages + messages)

        last = self._conversation.messages[-1] if self._conversation.messages else None
        first_new = messages[0] if messages else None
        if first_new and (
            not last
            or last.role != first_new.get("role")
            or last.content != first_new.get("content")
        ):
            for msg in self._normalize_message(messages):
                self._conversation.messages.append(msg)

        return effective_messages

    def _merge_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge consecutive same-role messages.

        Anthropic requires alternating user/assistant roles. The STT handler
        may add the user message before simple_response does, producing
        consecutive same-role entries. This merges them so no content is lost.
        """
        merged: list[dict[str, Any]] = []
        for msg in messages:
            if merged and msg.get("role") == merged[-1].get("role"):
                prev = merged[-1].get("content", "")
                curr = msg.get("content", "")
                if prev == curr:
                    merged[-1] = msg
                else:
                    prev_blocks = (
                        prev
                        if isinstance(prev, list)
                        else [{"type": "text", "text": str(prev)}]
                    )
                    curr_blocks = (
                        curr
                        if isinstance(curr, list)
                        else [{"type": "text", "text": str(curr)}]
                    )
                    merged[-1] = {**msg, "content": prev_blocks + curr_blocks}
            else:
                merged.append(msg)
        return merged

    def _convert_tools_to_provider_format(
        self, tools: List[ToolSchema]
    ) -> List[Dict[str, Any]]:
        """
        Convert ToolSchema objects to Anthropic format.

        Args:
            tools: List of ToolSchema objects

        Returns:
            List of tools in Anthropic format
        """
        anthropic_tools = []
        for tool in tools:
            anthropic_tool = {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool["parameters_schema"],
            }
            anthropic_tools.append(anthropic_tool)
        return anthropic_tools

    def _extract_tool_calls_from_response(
        self, response: Any
    ) -> List[NormalizedToolCallItem]:
        """
        Extract tool calls from Anthropic response.

        Args:
            response: Anthropic response object

        Returns:
            List of normalized tool call items
        """
        tool_calls = []

        if hasattr(response, "content") and response.content:
            for content_block in response.content:
                if hasattr(content_block, "type") and content_block.type == "tool_use":
                    tool_call: NormalizedToolCallItem = {
                        "type": "tool_call",
                        "id": content_block.id,  # Critical: capture the id for tool_result
                        "name": content_block.name,
                        "arguments_json": content_block.input
                        or {},  # normalize to arguments_json
                    }
                    tool_calls.append(tool_call)

        return tool_calls

    def _extract_tool_calls_from_stream_chunk(  # type: ignore[override]
        self,
        chunk: Any,
        current_tool_call: Optional[NormalizedToolCallItem] = None,
    ) -> tuple[List[NormalizedToolCallItem], Optional[NormalizedToolCallItem]]:
        """
        Extract tool calls from Anthropic streaming chunk using index-keyed accumulation.
        Args:
            chunk: Anthropic streaming event
            current_tool_call: Currently accumulating tool call (unused in this implementation)
        Returns:
            Tuple of (completed tool calls, current tool call being accumulated)
        """
        tool_calls = []
        t = getattr(chunk, "type", None)

        if t == "content_block_start":
            cb = getattr(chunk, "content_block", None)
            if getattr(cb, "type", None) == "tool_use":
                if cb is not None:
                    self._pending_tool_uses_by_index[chunk.index] = {
                        "id": cb.id,
                        "name": cb.name,
                        "parts": [],
                    }

        elif t == "content_block_delta":
            d = getattr(chunk, "delta", None)
            if getattr(d, "type", None) == "input_json_delta":
                pj = getattr(d, "partial_json", None)
                if pj is not None and chunk.index in self._pending_tool_uses_by_index:
                    self._pending_tool_uses_by_index[chunk.index]["parts"].append(pj)

        elif t == "content_block_stop":
            pending = self._pending_tool_uses_by_index.pop(chunk.index, None)
            if pending:
                buf = "".join(pending["parts"]).strip() or "{}"
                try:
                    args = json.loads(buf)
                except Exception:
                    args = {}
                tool_call_item: NormalizedToolCallItem = {
                    "type": "tool_call",
                    "id": pending["id"],
                    "name": pending["name"],
                    "arguments_json": args,
                }
                tool_calls.append(tool_call_item)
        return tool_calls, None

    def _create_tool_result_message(
        self, tool_calls: List[NormalizedToolCallItem], results: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Create tool result messages for Anthropic.
            tool_calls: List of tool calls that were executed
            results: List of results from function execution
        Returns:
            List of tool result messages in Anthropic format
        """
        # Create a single user message with tool_result blocks
        blocks = []
        for tool_call, result in zip(tool_calls, results):
            # Convert result to string if it's not already
            if isinstance(result, (str, int, float)):
                payload = str(result)
            else:
                payload = json.dumps(result)
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call["id"],  # Critical: must match tool_use.id
                    "content": payload,
                }
            )
        return [{"role": "user", "content": blocks}]

    def _concat_text_blocks(self, content):
        """Safely extract text from all text blocks in content."""
        out = []
        for b in content or []:
            if getattr(b, "type", None) == "text" and getattr(b, "text", None):
                out.append(b.text)
        return "".join(out)


def normalize_claude_messages(claude_messages: Any) -> List[Message]:
    if isinstance(claude_messages, str):
        claude_messages = [{"content": claude_messages, "role": "user", "type": "text"}]

    if not isinstance(claude_messages, (list, tuple)):
        claude_messages = [claude_messages]

    messages: List[Message] = []
    for m in claude_messages:
        if isinstance(m, dict):
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    item.get("text", "") for item in content if isinstance(item, dict)
                )
            role = m.get("role", "user")
        else:
            content = str(m)
            role = "user"
        messages.append(Message(original=m, content=content, role=role))

    return messages
