import json
import logging
import time
from typing import Any, AsyncIterator

from vision_agents.core.edge.types import Participant
from vision_agents.core.llm.llm import LLM, LLMResponseDelta, LLMResponseFinal
from vision_agents.core.llm.llm_types import NormalizedToolCallItem, ToolSchema
from xai_sdk import AsyncClient
from xai_sdk.aio.chat import Chat
from xai_sdk.chat import Response, system, tool, tool_result, user

logger = logging.getLogger(__name__)


class XAILLM(LLM):
    """
    The XAILLM class provides full/native access to the xAI SDK methods.
    It only standardizes the minimal feature set that's needed for the agent integration.

    The agent requires that we standardize:
    - sharing instructions
    - keeping conversation history
    - response normalization

    Notes on the xAI integration
    - simple_response maps to xAI chat.stream()
    - history is maintained using the chat object's append method

    Examples:

        from vision_agents.plugins import xai
        llm = xai.LLM(model="grok-4-latest")

    """

    provider_name = "xai"

    def __init__(
        self,
        model: str = "grok-4-latest",
        api_key: str | None = None,
        client: AsyncClient | None = None,
        tools_max_rounds: int = 3,
    ):
        """
        Initialize the XAILLM class.

        Args:
            model (str): The xAI model to use. Defaults to "grok-4-latest"
            api_key: optional API key. by default loads from XAI_API_KEY
            client: optional xAI client. by default creates a new client object.
            tools_max_rounds: max calling rounds for multi-hop tool call. Default - ``3``.
        """
        super().__init__()
        self.model = model
        self.xai_chat: Chat | None = None
        self._tools_max_rounds = max(tools_max_rounds, 1)

        if client is not None:
            self.client = client
        elif api_key is not None and api_key != "":
            self.client = AsyncClient(api_key=api_key)
        else:
            self.client = AsyncClient()

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
        tools = self._get_tools_for_provider()

        if not self.xai_chat:
            messages = []
            if self._instructions:
                messages.append(system(self._instructions))
            create_kwargs = {"model": self.model, "messages": messages}
            if tools:
                create_kwargs["tools"] = tools
            self.xai_chat = self.client.chat.create(**create_kwargs)

        assert self.xai_chat is not None
        self.xai_chat.append(user(text))

        request_start_time = time.perf_counter()
        first_token_time: float | None = None
        sequence_number = 0
        last_response: Response | None = None
        seen_calls: set[tuple[str | None, str, str]] = set()
        exec_seen: set[tuple[str | None, str, str]] = set()

        for round_num in range(self._tools_max_rounds + 1):
            pending_tool_calls: list[NormalizedToolCallItem] = []

            try:
                stream = self.xai_chat.stream()
            except Exception as e:
                logger.exception(
                    f'Failed to get a response from the LLM "{self.model}"'
                )
                self.on_llm_error(error=e)
                yield LLMResponseFinal(original=None, text="")
                return

            async for response, chunk in stream:
                last_response = response

                if chunk.content:
                    is_first_chunk = first_token_time is None
                    if is_first_chunk:
                        first_token_time = time.perf_counter()
                    delta_ttft_ms = (
                        (first_token_time - request_start_time) * 1000
                        if is_first_chunk and first_token_time is not None
                        else None
                    )
                    yield LLMResponseDelta(
                        content_index=0,
                        item_id=response.id,
                        output_index=0,
                        sequence_number=sequence_number,
                        delta=chunk.content,
                        is_first_chunk=is_first_chunk,
                        time_to_first_token_ms=delta_ttft_ms,
                    )
                    sequence_number += 1

                if chunk.choices and chunk.choices[0].finish_reason:
                    calls = self._extract_tool_calls_from_response(response)
                    for call in calls:
                        key = self._tc_key(call)
                        if key not in seen_calls:
                            pending_tool_calls.append(call)
                            seen_calls.add(key)

            if last_response is not None:
                self.xai_chat.append(last_response)

            if not pending_tool_calls or round_num >= self._tools_max_rounds:
                break

            triples, exec_seen = await self._dedup_and_execute(
                pending_tool_calls,
                max_concurrency=8,
                timeout_s=30,
                seen=exec_seen,
            )
            tool_messages = self._build_tool_result_messages(triples)
            if not tool_messages:
                break

            for message in tool_messages:
                self.xai_chat.append(message)

        latency_ms = (time.perf_counter() - request_start_time) * 1000
        ttft_ms: float | None = None
        if first_token_time is not None:
            ttft_ms = (first_token_time - request_start_time) * 1000

        text_out = ""
        item_id: str | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None
        total_tokens: int | None = None
        if last_response is not None:
            text_out = last_response.content
            item_id = last_response.id
            usage = last_response.usage
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens

        yield LLMResponseFinal(
            original=last_response,
            text=text_out,
            item_id=item_id,
            latency_ms=latency_ms,
            time_to_first_token_ms=ttft_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            model=self.model,
        )

    def _build_tool_result_messages(
        self, triples: list[tuple[NormalizedToolCallItem, Any, Any]]
    ) -> list[Any]:
        tool_messages = []
        for tc, res, err in triples:
            call_id = tc.get("id")
            if not call_id:
                continue

            output = err if err is not None else res
            output_str = self._sanitize_tool_output(output)
            tool_messages.append(tool_result(output_str, tool_call_id=call_id))
        return tool_messages

    def _convert_tools_to_provider_format(self, tools: list[ToolSchema]) -> list[Any]:
        """
        Convert ToolSchema objects to xAI SDK format.

        Args:
            tools: list of ToolSchema objects from the function registry

        Returns:
            list of tool objects in xAI SDK format
        """
        out = []
        for t in tools or []:
            if not isinstance(t, dict):
                continue
            name = t.get("name", "unnamed_tool")
            description = t.get("description", "") or ""
            params = t.get("parameters_schema") or t.get("parameters") or {}
            if not isinstance(params, dict):
                params = {}
            params.setdefault("type", "object")
            params.setdefault("properties", {})
            params.setdefault("additionalProperties", False)

            out.append(
                tool(
                    name=name,
                    description=description,
                    parameters=params,
                )
            )
        return out

    def _extract_tool_calls_from_response(
        self, response: Response
    ) -> list[NormalizedToolCallItem]:
        """Extract tool calls from xAI response.

        Args:
            response: xAI Response object

        Returns:
            list of normalized tool call items
        """
        calls = []
        for tc in response.tool_calls:
            func = tc.function
            name = func.name or "unknown"
            args_str = func.arguments or "{}"

            try:
                args_obj = json.loads(args_str)
            except json.JSONDecodeError:
                args_obj = {}

            call_item: NormalizedToolCallItem = {
                "type": "tool_call",
                "id": tc.id,
                "name": name,
                "arguments_json": args_obj,
            }
            calls.append(call_item)
        return calls
