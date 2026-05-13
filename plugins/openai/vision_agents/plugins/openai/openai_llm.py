import json
import logging
import time
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional

from openai import APIError, AsyncOpenAI
from openai.types.responses import Response as OpenAIResponse
from openai.types.responses import ResponseFunctionToolCall
from vision_agents.core.edge.types import Participant
from vision_agents.core.llm.llm import LLM, LLMResponseDelta, LLMResponseFinal
from vision_agents.core.llm.llm_types import NormalizedToolCallItem, ToolSchema

from .tool_utils import (
    convert_tools_to_openai_format,
    parse_tool_arguments,
    tool_call_dedup_key,
)

if TYPE_CHECKING:
    from vision_agents.core.agents.conversation import Message

logger = logging.getLogger(__name__)


class OpenAILLM(LLM):
    """
    The OpenAILLM class provides full/native access to the openAI SDK methods.
    It only standardized the minimal feature set that's needed for the agent integration.

    The agent requires that we standardize:
    - sharing instructions
    - keeping conversation history
    - response normalization

    Notes on the OpenAI integration
    - history is maintained using conversation.create()

    Examples:

        from vision_agents.plugins import openai
        llm = openai.LLM(model="gpt-5")

    """

    provider_name = "openai"

    def __init__(
        self,
        model: str = "gpt-5.4",
        api_key: str | None = None,
        base_url: str | None = None,
        client: AsyncOpenAI | None = None,
        max_tool_rounds: int = 3,
    ):
        """
        Initialize the OpenAILLM class.

        Args:
            model: The OpenAI model to use. https://platform.openai.com/docs/models
            api_key: Optional API key. By default loads from OPENAI_API_KEY.
            base_url: Optional base URL for the API.
            client: Optional OpenAI client. By default creates a new client object.
            max_tool_rounds: Maximum number of tool calling rounds (default 3).
        """
        super().__init__()
        self.model = model
        self.max_tool_rounds = max_tool_rounds
        self.openai_conversation: Optional[Any] = None
        self.conversation = None

        if client is not None:
            self.client = client
        elif api_key:
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = AsyncOpenAI(base_url=base_url)

    async def simple_response(
        self,
        text: str,
        participant: Participant | None = None,
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        """
        simple_response is a standardized way (across openai, claude, gemini etc.) to create a response.

        Args:
            text: The text to respond to
            participant: optionally the participant object

        Examples:

            llm.simple_response("say hi to the user, be mean")
        """
        await self._create_conversation()

        base_kwargs: Dict[str, Any] = {
            "model": self.model,
            "stream": True,
        }
        if self.openai_conversation:
            base_kwargs["conversation"] = self.openai_conversation.id
        tools = self.get_available_functions()
        if tools:
            base_kwargs["tools"] = convert_tools_to_openai_format(tools)
        if self._instructions:
            base_kwargs["instructions"] = self._instructions

        request_start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        sequence_number = 0
        last_response: Optional[OpenAIResponse] = None
        seen_calls: set[tuple[str, str]] = set()
        exec_seen: set[tuple[Optional[str], str, str]] = set()

        current_input: Any = text
        for round_num in range(self.max_tool_rounds + 1):
            new_tool_calls: list[NormalizedToolCallItem] = []
            try:
                stream = await self.client.responses.create(
                    **base_kwargs, input=current_input
                )
                async for event in stream:
                    if event.type == "response.failed":
                        error = event.response.error if event.response else None
                        error_message = error.message if error else "Unknown error"
                        error_code = error.code if error else "unknown"
                        logger.error(
                            "OpenAI stream error: %s (code=%s)",
                            error_message,
                            error_code,
                        )
                        self.on_llm_error(
                            error_type="response.failed",
                            error_code=error_code,
                        )
                        break
                    elif event.type == "response.output_text.delta":
                        is_first = first_token_time is None
                        if is_first:
                            first_token_time = time.perf_counter()
                        ttft_ms = (
                            (first_token_time - request_start_time) * 1000
                            if is_first and first_token_time is not None
                            else None
                        )
                        yield LLMResponseDelta(
                            delta=event.delta,
                            item_id=event.item_id,
                            output_index=event.output_index,
                            sequence_number=sequence_number,
                            is_first_chunk=is_first,
                            time_to_first_token_ms=ttft_ms,
                        )
                        sequence_number += 1
                    elif event.type == "response.completed":
                        last_response = event.response
                        for c in self._extract_tool_calls_from_response(event.response):
                            key = tool_call_dedup_key(c)
                            if key not in seen_calls:
                                new_tool_calls.append(c)
                                seen_calls.add(key)
            except APIError as e:
                logger.exception(
                    f'Failed to get a response from the LLM "{self.model}"'
                )
                self.on_llm_error(error=e)
                break

            if not new_tool_calls or round_num >= self.max_tool_rounds:
                break

            triples, exec_seen = await self._dedup_and_execute(
                new_tool_calls,
                max_concurrency=8,
                timeout_s=30,
                seen=exec_seen,
            )
            tool_messages = self._build_tool_messages(triples)
            if not tool_messages:
                break
            current_input = tool_messages

        latency_ms = (time.perf_counter() - request_start_time) * 1000
        ttft_final_ms: Optional[float] = None
        if first_token_time is not None:
            ttft_final_ms = (first_token_time - request_start_time) * 1000

        text_out = ""
        item_id: Optional[str] = None
        input_tokens: Optional[int] = None
        output_tokens: Optional[int] = None
        total_tokens: Optional[int] = None
        model = self.model
        if last_response is not None:
            text_out = last_response.output_text
            item_id = last_response.output[0].id if last_response.output else None
            if last_response.usage:
                input_tokens = last_response.usage.input_tokens
                output_tokens = last_response.usage.output_tokens
                total_tokens = last_response.usage.total_tokens
            model = last_response.model

        yield LLMResponseFinal(
            text=text_out,
            item_id=item_id,
            latency_ms=latency_ms,
            time_to_first_token_ms=ttft_final_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            model=model,
            original=last_response,
        )

    async def close(self) -> None:
        await self.client.close()

    async def _create_conversation(self):
        if not self.openai_conversation:
            self.openai_conversation = await self.client.conversations.create()

    def _build_tool_messages(
        self, triples: list[tuple[dict[str, Any], Any, Any]]
    ) -> list[dict[str, Any]]:
        """Build tool result messages from execution results."""
        tool_messages = []
        for tc, res, err in triples:
            call_id = tc.get("id")
            if not call_id:
                continue

            output = err if err is not None else res
            output_str = self._sanitize_tool_output(output)
            tool_messages.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output_str,
                }
            )
        return tool_messages

    @staticmethod
    def _normalize_message(openai_input) -> List["Message"]:
        """
        Takes the openAI list of messages and standardizes it so we can store it in chat
        """
        from vision_agents.core.agents.conversation import Message

        # standardize on input
        if isinstance(openai_input, str):
            openai_input = [dict(content=openai_input, role="user", type="message")]
        elif not isinstance(openai_input, List):
            openai_input = [openai_input]

        messages: List[Message] = []
        for i in openai_input:
            content = i.get("content", i if isinstance(i, str) else json.dumps(i))
            message = Message(original=i, content=content)
            messages.append(message)

        return messages

    def _convert_tools_to_provider_format(
        self, tools: List[ToolSchema]
    ) -> List[Dict[str, Any]]:
        """Convert ToolSchema objects to OpenAI format."""
        return convert_tools_to_openai_format(tools)

    def _extract_tool_calls_from_response(
        self, response: OpenAIResponse
    ) -> List[NormalizedToolCallItem]:
        """Extract tool calls from OpenAI response."""
        calls: List[NormalizedToolCallItem] = []
        for item in response.output or []:
            if isinstance(item, ResponseFunctionToolCall):
                calls.append(
                    {
                        "type": "tool_call",
                        "id": item.call_id,
                        "name": item.name,
                        "arguments_json": parse_tool_arguments(item.arguments),
                    }
                )
        return calls
