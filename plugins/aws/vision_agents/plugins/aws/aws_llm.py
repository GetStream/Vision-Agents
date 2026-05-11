import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator

import boto3
from botocore.exceptions import ClientError
from vision_agents.core.edge.types import Participant
from vision_agents.core.llm.llm import LLM, LLMResponseDelta, LLMResponseFinal
from vision_agents.core.llm.llm_types import NormalizedToolCallItem, ToolSchema

logger = logging.getLogger(__name__)


PLUGIN_NAME = "aws"


_STREAM_SENTINEL: Any = object()


async def _iter_stream(
    client, request_kwargs: dict[str, Any]
) -> AsyncIterator[dict[str, Any]]:
    """Yield Bedrock converse_stream events one at a time from a worker thread."""
    response = await asyncio.to_thread(client.converse_stream, **request_kwargs)
    stream = response.get("stream")
    if not stream:
        return
    iterator = iter(stream)
    while True:
        event = await asyncio.to_thread(next, iterator, _STREAM_SENTINEL)
        if event is _STREAM_SENTINEL:
            return
        yield event


class BedrockLLM(LLM):
    """
    AWS Bedrock LLM integration for Vision Agents.

    Converse docs can be found here:
    https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html

    Chat history has to be manually passed, there is no conversation storage.

    Examples:

        from vision_agents.plugins import aws
        llm = aws.LLM(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            region_name="us-east-1"
        )
    """

    provider_name = PLUGIN_NAME

    def __init__(
        self,
        model: str,
        region_name: str = "us-east-1",
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        aws_profile: str | None = None,
        tools_max_rounds: int = 3,
    ):
        """
        Initialize the BedrockLLM class.

        Args:
            model: The Bedrock model ID (e.g., "anthropic.claude-3-5-sonnet-20241022-v2:0")
            region_name: AWS region name (default: "us-east-1")
            aws_access_key_id: Optional AWS access key ID
            aws_secret_access_key: Optional AWS secret access key
            aws_session_token: Optional AWS session token
            aws_profile: Optional AWS profile name (from ~/.aws/credentials or ~/.aws/config)
            tools_max_rounds: max calling rounds for multi-hop tool call. Default - ``3``.
        """
        super().__init__()
        self.model = model
        self._pending_tool_uses_by_index: dict[int, dict[str, Any]] = {}

        # Build boto3 Session kwargs
        session_kwargs: dict[str, Any] = {"region_name": region_name}
        if aws_profile:
            session_kwargs["profile_name"] = aws_profile
        if aws_access_key_id:
            session_kwargs["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            session_kwargs["aws_secret_access_key"] = aws_secret_access_key
        if aws_session_token:
            session_kwargs["aws_session_token"] = aws_session_token

        self._client = None
        self._session_kwargs = session_kwargs
        self._tools_max_rounds = max(tools_max_rounds, 1)
        self.region_name = region_name

    @property
    async def client(self) -> Any:
        if self._client is None:

            def _create_client():
                session = boto3.Session(**self._session_kwargs)
                self._client = session.client("bedrock-runtime")

            await asyncio.to_thread(_create_client)
        return self._client

    async def simple_response(
        self,
        text: str,
        participant: Participant | None = None,
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        """
        simple_response is a standardized way to create an LLM response.

        This method is also called every time the new STT transcript is received.

        Args:
            text: The text to respond to.
            participant: the Participant object, optional.

        Examples:

            llm.simple_response("say hi to the user, be nice")
        """
        if self._conversation is None:
            logger.warning(
                f'Cannot request a response from the LLM "{self.model}" - the conversation has not been initialized yet.'
            )
            yield LLMResponseFinal(original=None, text="")
            return

        messages = await self._build_model_request(text)
        tools_param: list[dict[str, Any]] | None = None
        tools_spec = self.get_available_functions()
        if tools_spec:
            tools_param = self._convert_tools_to_provider_format(tools_spec)

        async for item in self._create_response_internal(
            messages=messages, tools=tools_param
        ):
            yield item

    async def _build_model_request(self, text: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self._conversation is not None:
            for message in self._conversation.messages:
                content = (message.content or "").strip()
                if not content:
                    continue
                messages.append(
                    {
                        "role": message.role,
                        "content": [{"text": message.content}],
                    }
                )
        new_user = {"role": "user", "content": [{"text": text}]}
        if messages and messages[-1].get("role") == "user":
            messages[-1] = new_user
        else:
            messages.append(new_user)
        return messages

    async def _create_response_internal(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        """Internal method to create response from a streamed converse_stream call."""
        request_kwargs: dict[str, Any] = {
            "modelId": self.model,
            "messages": messages,
        }
        if self._instructions:
            request_kwargs["system"] = [{"text": self._instructions}]
        if tools:
            request_kwargs["toolConfig"] = {"tools": tools}

        request_start_time = time.perf_counter()
        client = await self.client
        # Bedrock does not provide any response ids to match the chunks together.
        # Generate our own so the conversation collapses deltas into one message.
        response_id = str(uuid.uuid4())

        text_chunks: list[str] = []
        self._pending_tool_uses_by_index.clear()
        accumulated_tool_calls: list[NormalizedToolCallItem] = []
        sequence_number = 0
        first_token_time: float | None = None
        last_event: dict[str, Any] | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None

        try:
            async for event in _iter_stream(client, request_kwargs):
                last_event = event

                if "contentBlockStart" in event:
                    start = event["contentBlockStart"].get("start", {})
                    if "toolUse" in start:
                        tool_use = start["toolUse"]
                        idx = event["contentBlockStart"].get("contentBlockIndex", 0)
                        self._pending_tool_uses_by_index[idx] = {
                            "id": tool_use.get("toolUseId", ""),
                            "name": tool_use.get("name", ""),
                            "parts": [],
                        }

                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta", {})
                    if delta.get("text"):
                        is_first = first_token_time is None
                        ttft_ms: float | None = None
                        if is_first:
                            first_token_time = time.perf_counter()
                            ttft_ms = (first_token_time - request_start_time) * 1000
                        text_chunks.append(delta["text"])
                        yield LLMResponseDelta(
                            content_index=event["contentBlockDelta"].get(
                                "contentBlockIndex", 0
                            ),
                            item_id=response_id,
                            output_index=0,
                            sequence_number=sequence_number,
                            delta=delta["text"],
                            is_first_chunk=is_first,
                            time_to_first_token_ms=ttft_ms,
                        )
                        sequence_number += 1
                    if "toolUse" in delta:
                        idx = event["contentBlockDelta"].get("contentBlockIndex", 0)
                        if idx in self._pending_tool_uses_by_index:
                            input_data = delta["toolUse"].get("input", "")
                            self._pending_tool_uses_by_index[idx]["parts"].append(
                                input_data
                            )

                if "contentBlockStop" in event:
                    idx = event["contentBlockStop"].get("contentBlockIndex", 0)
                    pending = self._pending_tool_uses_by_index.pop(idx, None)
                    if pending:
                        buf = "".join(pending["parts"]).strip() or "{}"
                        try:
                            args = json.loads(buf)
                        except json.JSONDecodeError:
                            args = {}
                        tool_call: NormalizedToolCallItem = {
                            "type": "tool_call",
                            "id": pending["id"],
                            "name": pending["name"],
                            "arguments_json": args,
                        }
                        accumulated_tool_calls.append(tool_call)

                if "metadata" in event:
                    usage = event["metadata"].get("usage", {})
                    if usage:
                        input_tokens = usage.get("inputTokens")
                        output_tokens = usage.get("outputTokens")
        except ClientError as e:
            logger.exception(f'Failed to get a response from the LLM "{self.model}"')
            self.metrics.on_llm_error(
                provider=self.provider_name,
                error_type=type(e).__name__,
            )
            yield LLMResponseFinal(original=None, text="")
            return

        total_text = "".join(text_chunks)

        if accumulated_tool_calls:
            async for item in self._handle_tool_calls(
                accumulated_tool_calls,
                messages,
                tools,
                request_start_time,
                first_token_time,
                sequence_number,
                response_id,
            ):
                yield item
            return

        latency_ms = (time.perf_counter() - request_start_time) * 1000
        ttft_total: float | None = None
        if first_token_time is not None:
            ttft_total = (first_token_time - request_start_time) * 1000

        total_tokens: int | None = None
        if input_tokens is not None or output_tokens is not None:
            total_tokens = (input_tokens or 0) + (output_tokens or 0)

        yield LLMResponseFinal(
            original=last_event,
            text=total_text,
            item_id=response_id,
            latency_ms=latency_ms,
            time_to_first_token_ms=ttft_total,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            model=self.model,
        )

    async def _handle_tool_calls(
        self,
        tool_calls: list[NormalizedToolCallItem],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        request_start_time: float,
        first_token_time: float | None,
        sequence_number: int,
        response_id: str,
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        """Execute tool calls and stream follow-up responses for up to _tools_max_rounds rounds."""
        current_tool_calls = tool_calls
        seen: set[tuple] = set()
        current_messages = list(messages)
        last_event: dict[str, Any] | None = None
        total_text = ""
        input_tokens: int | None = None
        output_tokens: int | None = None

        client = await self.client

        for round_num in range(self._tools_max_rounds):
            triples, seen = await self._dedup_and_execute(
                current_tool_calls,
                max_concurrency=8,
                timeout_s=30,
                seen=seen,
            )

            if not triples:
                break

            assistant_content: list[dict[str, Any]] = []
            tool_result_content: list[dict[str, Any]] = []
            for tc, res, err in triples:
                cid = tc.get("id")
                if not cid:
                    continue

                assistant_content.append(
                    {
                        "toolUse": {
                            "toolUseId": cid,
                            "name": tc["name"],
                            "input": tc.get("arguments_json", {}),
                        }
                    }
                )
                tool_result_content.append(
                    {
                        "toolResult": {
                            "toolUseId": cid,
                            "content": [
                                {
                                    "text": self._sanitize_tool_output(
                                        err if err is not None else res
                                    )
                                }
                            ],
                        }
                    }
                )

            if not tool_result_content:
                break

            current_messages.append({"role": "assistant", "content": assistant_content})
            current_messages.append({"role": "user", "content": tool_result_content})

            request_kwargs: dict[str, Any] = {
                "modelId": self.model,
                "messages": current_messages,
            }
            if self._instructions:
                request_kwargs["system"] = [{"text": self._instructions}]
            if tools:
                request_kwargs["toolConfig"] = {"tools": tools}

            text_chunks: list[str] = []
            self._pending_tool_uses_by_index.clear()
            next_tool_calls: list[NormalizedToolCallItem] = []

            try:
                async for event in _iter_stream(client, request_kwargs):
                    last_event = event

                    if "contentBlockStart" in event:
                        start = event["contentBlockStart"].get("start", {})
                        if "toolUse" in start:
                            tool_use = start["toolUse"]
                            idx = event["contentBlockStart"].get("contentBlockIndex", 0)
                            self._pending_tool_uses_by_index[idx] = {
                                "id": tool_use.get("toolUseId", ""),
                                "name": tool_use.get("name", ""),
                                "parts": [],
                            }

                    if "contentBlockDelta" in event:
                        delta = event["contentBlockDelta"].get("delta", {})
                        if delta.get("text"):
                            is_first = first_token_time is None
                            ttft_ms: float | None = None
                            if is_first:
                                first_token_time = time.perf_counter()
                                ttft_ms = (first_token_time - request_start_time) * 1000

                            text_chunks.append(delta["text"])
                            yield LLMResponseDelta(
                                content_index=event["contentBlockDelta"].get(
                                    "contentBlockIndex", 0
                                ),
                                item_id=response_id,
                                output_index=0,
                                sequence_number=sequence_number,
                                delta=delta["text"],
                                is_first_chunk=is_first,
                                time_to_first_token_ms=ttft_ms,
                            )
                            sequence_number += 1
                        if "toolUse" in delta:
                            idx = event["contentBlockDelta"].get("contentBlockIndex", 0)
                            if idx in self._pending_tool_uses_by_index:
                                input_data = delta["toolUse"].get("input", "")
                                self._pending_tool_uses_by_index[idx]["parts"].append(
                                    input_data
                                )

                    if "contentBlockStop" in event:
                        idx = event["contentBlockStop"].get("contentBlockIndex", 0)
                        pending = self._pending_tool_uses_by_index.pop(idx, None)
                        if pending:
                            buf = "".join(pending["parts"]).strip() or "{}"
                            try:
                                args = json.loads(buf)
                            except json.JSONDecodeError:
                                args = {}
                            tool_call: NormalizedToolCallItem = {
                                "type": "tool_call",
                                "id": pending["id"],
                                "name": pending["name"],
                                "arguments_json": args,
                            }
                            next_tool_calls.append(tool_call)

                    if "metadata" in event:
                        usage = event["metadata"].get("usage", {})
                        if usage:
                            input_tokens = usage.get("inputTokens")
                            output_tokens = usage.get("outputTokens")
            except ClientError as e:
                logger.exception("Failed to get follow-up response")
                self.metrics.on_llm_error(
                    provider=self.provider_name,
                    error_type=type(e).__name__,
                )
                break

            total_text = "".join(text_chunks)

            if next_tool_calls and round_num < self._tools_max_rounds - 1:
                current_tool_calls = next_tool_calls
                continue

            break

        latency_ms = (time.perf_counter() - request_start_time) * 1000
        ttft_total: float | None = None
        if first_token_time is not None:
            ttft_total = (first_token_time - request_start_time) * 1000

        total_tokens: int | None = None
        if input_tokens is not None or output_tokens is not None:
            total_tokens = (input_tokens or 0) + (output_tokens or 0)

        yield LLMResponseFinal(
            original=last_event,
            text=total_text,
            item_id=response_id,
            latency_ms=latency_ms,
            time_to_first_token_ms=ttft_total,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            model=self.model,
        )

    def _convert_tools_to_provider_format(
        self, tools: list[ToolSchema]
    ) -> list[dict[str, Any]]:
        """
        Convert ToolSchema objects to AWS Bedrock format.

        Args:
            tools: List of ToolSchema objects

        Returns:
            List of tools in AWS Bedrock format
        """
        aws_tools = []
        for tool in tools:
            name = tool.get("name", "unnamed_tool")
            description = tool.get("description", "") or ""
            params = tool.get("parameters_schema") or {}

            # Normalize to a valid JSON Schema object
            if not isinstance(params, dict):
                params = {}

            # Ensure it has the required JSON Schema structure
            if "type" not in params:
                # Extract required fields from properties if they exist
                properties = params if params else {}
                required = list(properties.keys()) if properties else []

                params = {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                }
            else:
                # Already has type, but ensure additionalProperties is set
                if "additionalProperties" not in params:
                    params["additionalProperties"] = False

            aws_tool = {
                "toolSpec": {
                    "name": name,
                    "description": description,
                    "inputSchema": {
                        "json": params  # This is a dict, not a JSON string
                    },
                }
            }
            aws_tools.append(aws_tool)
        return aws_tools
