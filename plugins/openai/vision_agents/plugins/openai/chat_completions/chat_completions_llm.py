import json
import logging
import time
from typing import Any, AsyncIterator, Optional, cast

from openai import AsyncOpenAI, AsyncStream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from vision_agents.core.edge.types import Participant
from vision_agents.core.llm.llm import LLM, LLMResponseDelta, LLMResponseFinal
from vision_agents.core.llm.llm_types import NormalizedToolCallItem, ToolSchema

logger = logging.getLogger(__name__)


PLUGIN_NAME = "chat_completions_llm"


class ChatCompletionsLLM(LLM):
    """
    This plugin allows developers to easily interact with models that use Chat Completions API.
    The model is expected to accept text and respond with text.

    Features:
        - Streaming responses: Supports streaming text responses with real-time chunk events
        - Function calling: Supports tool/function calling with automatic execution
        - Event-driven: Emits LLM events (chunks, completion, errors) for integration with other components

    Examples:

        from vision_agents.plugins import openai
        llm = openai.ChatCompletionsLLM(model="deepseek-ai/DeepSeek-V3.1")

    """

    provider_name = PLUGIN_NAME

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Optional[AsyncOpenAI] = None,
        tools_max_rounds: int = 3,
    ):
        """
        Initialize the ChatCompletionsLLM class.

        Args:
            model (str): The model id to use.
            api_key: optional API key. By default, loads from OPENAI_API_KEY environment variable.
            base_url: optional base url. By default, loads from OPENAI_BASE_URL environment variable.
            client: optional `AsyncOpenAI` client. By default, creates a new client object.
            tools_max_rounds: max calling rounds for multi-hop tool call. Default - ``3``.
        """
        super().__init__()
        self.model = model
        self._tools_max_rounds = max(tools_max_rounds, 1)
        # Track tool calls being accumulated during streaming
        self._pending_tool_calls: dict[int, dict[str, Any]] = {}

        if client is not None:
            self._client = client
        else:
            self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def simple_response(
        self,
        text: str,
        participant: Optional[Participant] = None,
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        """
        simple_response is a standardized way to create an LLM response.

        This method is also called every time the new STT transcript is received.

        Args:
            text: The text to respond to.
            participant: the Participant object, optional. If not provided, the message will be sent with the "user" role.

        Examples:

            llm.simple_response("say hi to the user, be nice")
        """

        if self._conversation is None:
            # The agent hasn't joined the call yet.
            logger.warning(
                f'Cannot request a response from the LLM "{self.model}" - the conversation has not been initialized yet.'
            )
            yield LLMResponseFinal(original=None, text="")
            return

        # The simple_response is called directly without providing the participant -
        # assuming it's an initial prompt.
        if participant is None:
            await self._conversation.send_message(
                role="user", user_id="user", content=text
            )

        messages = await self._build_model_request()
        tools_param = None
        tools_spec = self.get_available_functions()
        if tools_spec:
            tools_param = self._convert_tools_to_provider_format(tools_spec)

        async for item in self._create_response_internal(
            messages=messages, tools=tools_param
        ):
            yield item

    async def close(self) -> None:
        await self._client.close()

    async def _create_response_internal(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        """Internal method to create response with tool handling loop."""
        request_kwargs: dict[str, Any] = {
            "messages": messages,
            "model": kwargs.get("model", self.model),
            "stream": True,
        }
        if tools:
            request_kwargs["tools"] = tools

        # Track timing
        request_start_time = time.perf_counter()

        try:
            response = await self._client.chat.completions.create(**request_kwargs)
        except Exception as e:
            logger.exception(f'Failed to get a response from the LLM "{self.model}"')
            self.metrics.on_llm_error(
                provider=self.provider_name,
                error_type=type(e).__name__,
            )
            yield LLMResponseFinal(original=None, text="")
            return

        async for item in self._process_streaming_response(
            response, messages, tools, kwargs, request_start_time
        ):
            yield item

    async def _process_streaming_response(
        self,
        response: Any,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        kwargs: dict[str, Any],
        request_start_time: float,
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        """Process a streaming response, handling tool calls if present."""
        text_chunks: list[str] = []
        self._pending_tool_calls = {}
        accumulated_tool_calls: list[NormalizedToolCallItem] = []
        sequence_number = 0
        first_token_time: Optional[float] = None
        last_chunk: ChatCompletionChunk | None = None

        async for chunk in cast(AsyncStream[ChatCompletionChunk], response):
            last_chunk = chunk
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            content = choice.delta.content
            finish_reason = choice.finish_reason

            if choice.delta.tool_calls:
                for tc in choice.delta.tool_calls:
                    self._accumulate_tool_call_chunk(tc)

            if content:
                is_first = first_token_time is None
                ttft_ms = None
                if is_first:
                    first_token_time = time.perf_counter()
                    ttft_ms = (first_token_time - request_start_time) * 1000

                text_chunks.append(content)
                yield LLMResponseDelta(
                    content_index=None,
                    item_id=chunk.id,
                    output_index=0,
                    sequence_number=sequence_number,
                    delta=content,
                    is_first_chunk=is_first,
                    time_to_first_token_ms=ttft_ms,
                )
                sequence_number += 1

            if finish_reason:
                if finish_reason in ("length", "content"):
                    logger.warning(
                        f'The model finished the response due to reason "{finish_reason}"'
                    )

                if finish_reason == "tool_calls":
                    accumulated_tool_calls = self._finalize_pending_tool_calls()

        total_text = "".join(text_chunks)

        # Handle tool calls if any were accumulated
        if accumulated_tool_calls:
            async for item in self._handle_tool_calls(
                accumulated_tool_calls,
                messages,
                tools,
                kwargs,
                request_start_time,
                first_token_time,
                sequence_number,
            ):
                yield item
            return

        latency_ms = (time.perf_counter() - request_start_time) * 1000
        ttft_ms = None
        if first_token_time is not None:
            ttft_ms = (first_token_time - request_start_time) * 1000

        item_id = last_chunk.id if last_chunk is not None else None
        yield LLMResponseFinal(
            original=last_chunk,
            text=total_text,
            item_id=item_id,
            latency_ms=latency_ms,
            time_to_first_token_ms=ttft_ms,
            model=self.model,
        )

    async def _build_model_request(self) -> list[dict]:
        messages: list[dict] = []
        # Add Agent's instructions as system prompt.
        if self._instructions:
            messages.append({"role": "system", "content": self._instructions})

        # Add all messages from the conversation to the prompt
        if self._conversation is not None:
            for message in self._conversation.messages:
                messages.append({"role": message.role, "content": message.content})
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

    def _finalize_pending_tool_calls(self) -> list[NormalizedToolCallItem]:
        """Convert accumulated tool call chunks into normalized tool calls."""
        tool_calls: list[NormalizedToolCallItem] = []
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
        self, tools: list[ToolSchema]
    ) -> list[dict[str, Any]]:
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

    async def _handle_tool_calls(
        self,
        tool_calls: list[NormalizedToolCallItem],
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        kwargs: dict[str, Any],
        request_start_time: float,
        first_token_time: Optional[float],
        sequence_number: int,
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        """Execute tool calls and get follow-up response."""
        current_tool_calls = tool_calls
        seen: set[tuple] = set()
        current_messages = list(messages)
        last_chunk: ChatCompletionChunk | None = None
        total_text = ""

        for round_num in range(self._tools_max_rounds):
            triples, seen = await self._dedup_and_execute(
                current_tool_calls,
                max_concurrency=8,
                timeout_s=30,
                seen=seen,
            )

            if not triples:
                break

            # Build assistant message with tool_calls
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
                break

            # Add assistant message with tool_calls, then tool results
            current_messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": assistant_tool_calls,
                }
            )
            current_messages.extend(tool_results)

            # Make follow-up request
            request_kwargs: dict[str, Any] = {
                "messages": current_messages,
                "model": kwargs.get("model", self.model),
                "stream": True,
            }
            if tools:
                request_kwargs["tools"] = tools

            try:
                follow_up = await self._client.chat.completions.create(**request_kwargs)
            except Exception:
                logger.exception("Failed to get follow-up response")
                break

            text_chunks: list[str] = []
            self._pending_tool_calls = {}
            next_tool_calls: list[NormalizedToolCallItem] = []

            async for chunk in cast(AsyncStream[ChatCompletionChunk], follow_up):
                last_chunk = chunk
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                content = choice.delta.content
                finish_reason = choice.finish_reason

                if choice.delta.tool_calls:
                    for tc in choice.delta.tool_calls:
                        self._accumulate_tool_call_chunk(tc)

                if content:
                    is_first = first_token_time is None
                    ttft_ms = None
                    if is_first:
                        first_token_time = time.perf_counter()
                        ttft_ms = (first_token_time - request_start_time) * 1000

                    text_chunks.append(content)
                    yield LLMResponseDelta(
                        content_index=None,
                        item_id=chunk.id,
                        output_index=0,
                        sequence_number=sequence_number,
                        delta=content,
                        is_first_chunk=is_first,
                        time_to_first_token_ms=ttft_ms,
                    )
                    sequence_number += 1

                if finish_reason:
                    if finish_reason == "tool_calls":
                        next_tool_calls = self._finalize_pending_tool_calls()

            total_text = "".join(text_chunks)

            # Continue if there are more tool calls
            if next_tool_calls and round_num < self._tools_max_rounds - 1:
                current_tool_calls = next_tool_calls
                continue

            break

        latency_ms = (time.perf_counter() - request_start_time) * 1000
        ttft_ms = None
        if first_token_time is not None:
            ttft_ms = (first_token_time - request_start_time) * 1000

        item_id = last_chunk.id if last_chunk is not None else None
        yield LLMResponseFinal(
            original=last_chunk,
            text=total_text,
            item_id=item_id,
            latency_ms=latency_ms,
            time_to_first_token_ms=ttft_ms,
            model=self.model,
        )
