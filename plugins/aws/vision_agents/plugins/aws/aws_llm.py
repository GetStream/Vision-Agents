import asyncio
import os
import logging
from typing import Optional, List, TYPE_CHECKING, Any, Dict, cast
import json
import boto3
from botocore.exceptions import ClientError

from vision_agents.core.llm.llm import LLM, LLMResponseEvent
from vision_agents.core.llm.llm_types import ToolSchema, NormalizedToolCallItem


from vision_agents.core.llm.events import (
    LLMResponseChunkEvent,
    LLMResponseCompletedEvent,
)
from vision_agents.core.processors import Processor
from . import events
from vision_agents.core.edge.types import Participant

if TYPE_CHECKING:
    from vision_agents.core.agents.conversation import Message


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

    def __init__(
        self,
        model: str,
        region_name: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
    ):
        """
        Initialize the BedrockLLM class.

        Args:
            model: The Bedrock model ID (e.g., "anthropic.claude-3-5-sonnet-20241022-v2:0")
            region_name: AWS region name (default: "us-east-1")
            aws_access_key_id: Optional AWS access key ID
            aws_secret_access_key: Optional AWS secret access key
            aws_session_token: Optional AWS session token
        """
        super().__init__()
        self.events.register_events_from_module(events)
        self.model = model
        self._pending_tool_uses_by_index: Dict[int, Dict[str, Any]] = {}

        # Initialize boto3 bedrock-runtime client
        session_kwargs = {"region_name": region_name}
        if aws_access_key_id:
            session_kwargs["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            session_kwargs["aws_secret_access_key"] = aws_secret_access_key
        if aws_session_token:
            session_kwargs["aws_session_token"] = aws_session_token

        if os.environ.get("AWS_BEDROCK_API_KEY"):
            session_kwargs["aws_session_token"] = os.environ["AWS_BEDROCK_API_KEY"]

        self._client = None
        self._session_kwargs = session_kwargs
        self.region_name = region_name
        self.logger = logging.getLogger(__name__)

    @property
    async def client(self) -> Any:
        if self._client is None:

            def _create_client():
                self._client = boto3.client("bedrock-runtime", **self._session_kwargs)

            await asyncio.to_thread(_create_client)
        return self._client

    async def simple_response(
        self,
        text: str,
        processors: Optional[List[Processor]] = None,
        participant: Optional[Participant] = None,
    ):
        """
        Simple response is a standardized way to create a response.

        Args:
            text: The text to respond to
            processors: list of processors (which contain state) about the video/voice AI
            participant: optionally the participant object

        Examples:

            await llm.simple_response("say hi to the user")
        """
        return await self.converse_stream(
            messages=[{"role": "user", "content": [{"text": text}]}]
        )

    async def converse(self, *args, **kwargs) -> LLMResponseEvent[Any]:
        """
        Converse gives full access to the Bedrock Converse API.
        This method wraps the Bedrock method and broadcasts events for agent integration.
        """
        if "modelId" not in kwargs:
            kwargs["modelId"] = self.model

        # Add tools if available
        tools = self.get_available_functions()
        if tools:
            converted_tools = self._convert_tools_to_provider_format(tools)
            kwargs["toolConfig"] = {"tools": converted_tools}

        # Combine original instructions with markdown file contents
        if self._instructions:
            kwargs["system"] = [{"text": self._instructions}]

        # Ensure the AI remembers the past conversation
        new_messages = kwargs.get("messages", [])
        if hasattr(self, "_conversation") and self._conversation:
            old_messages = [m.original for m in self._conversation.messages]
            kwargs["messages"] = old_messages + new_messages
            # Add messages to conversation
            normalized_messages = self._normalize_message(new_messages)
            for msg in normalized_messages:
                self._conversation.messages.append(msg)

        client = await self.client

        try:
            system_param = kwargs.get("system")

            response = await asyncio.to_thread(client.converse, **kwargs)

            # Extract text from response
            text = self._extract_text_from_response(response)
            llm_response = LLMResponseEvent(response, text)

            # Handle tool calls if present
            function_calls = self._extract_tool_calls_from_response(response)
            if function_calls:
                for i, fc in enumerate(function_calls):
                    self.logger.debug(
                        f"Tool call {i + 1}: {fc.get('name')} with args: {fc.get('arguments_json')}"
                    )
                messages = kwargs["messages"][:]
                assistant_msg_from_response = response.get("output", {}).get(
                    "message", {}
                )
                if assistant_msg_from_response:
                    messages.append(assistant_msg_from_response)

                MAX_ROUNDS = 3
                rounds = 0
                seen: set[tuple[str, str, str]] = set()
                current_calls = function_calls

                while current_calls and rounds < MAX_ROUNDS:
                    # Execute calls concurrently with dedup
                    triples, seen = await self._dedup_and_execute(
                        cast(List[NormalizedToolCallItem], current_calls),
                        seen=seen,
                        max_concurrency=8,
                        timeout_s=30,
                    )

                    if not triples:
                        self.logger.warning(
                            "No tool execution results despite tool calls"
                        )
                        break

                    # Build tool result message
                    tool_result_blocks = []
                    for tc, res, err in triples:
                        if err:
                            self.logger.error(
                                f"Tool {tc['name']} execution error: {err}"
                            )
                            tool_response = str(err)
                        else:
                            # Convert result to string format (AWS expects text, not json content type)
                            if isinstance(res, (dict, list)):
                                tool_response = json.dumps(res)
                            elif isinstance(res, str):
                                tool_response = res
                            else:
                                tool_response = str(res)

                        tool_result_blocks.append(
                            {
                                "toolUseId": tc["id"],
                                "content": [{"text": tool_response}],
                            }
                        )

                    user_tool_results_msg = {
                        "role": "user",
                        "content": [{"toolResult": tr} for tr in tool_result_blocks],
                    }
                    messages = messages + [user_tool_results_msg]
                    follow_up_kwargs = {
                        "modelId": self.model,
                        "messages": messages,
                        "toolConfig": kwargs.get("toolConfig", {}),
                    }
                    if system_param:
                        follow_up_kwargs["system"] = system_param

                    try:
                        follow_up_response = client.converse(**follow_up_kwargs)
                    except ClientError as e:
                        self.logger.error(
                            f"AWS Bedrock API error in follow-up call: {e}"
                        )
                        error_code = (
                            e.response.get("Error", {}).get("Code", "Unknown")
                            if hasattr(e, "response")
                            else "Unknown"
                        )
                        self.logger.error(
                            f"Error code: {error_code}, Full error: {str(e)}"
                        )
                        raise

                    current_calls = self._extract_tool_calls_from_response(
                        follow_up_response
                    )
                    follow_up_text = self._extract_text_from_response(
                        follow_up_response
                    )
                    llm_response = LLMResponseEvent(follow_up_response, follow_up_text)

                    if current_calls:
                        assistant_msg_from_follow_up = follow_up_response.get(
                            "output", {}
                        ).get("message", {})
                        if assistant_msg_from_follow_up:
                            messages.append(assistant_msg_from_follow_up)

                    if follow_up_text and not current_calls:
                        text = follow_up_text
                        break

                    rounds += 1

                if current_calls:
                    final_kwargs = {
                        "modelId": self.model,
                        "messages": messages,
                    }
                    if system_param:
                        final_kwargs["system"] = system_param

                    try:
                        final_response = client.converse(**final_kwargs)
                    except ClientError as e:
                        self.logger.error(f"AWS Bedrock API error in final pass: {e}")
                        error_code = (
                            e.response.get("Error", {}).get("Code", "Unknown")
                            if hasattr(e, "response")
                            else "Unknown"
                        )
                        self.logger.error(
                            f"Error code: {error_code}, Full error: {str(e)}"
                        )
                        raise

                    final_text = self._extract_text_from_response(final_response)
                    llm_response = LLMResponseEvent(final_response, final_text)
                    text = final_text
                elif rounds > 0:
                    text = llm_response.text or text

            final_text_for_event = llm_response.text or text
            original_for_event = llm_response.original or response

            if not final_text_for_event:
                self.logger.warning(
                    "Final response text is empty - model may not have responded"
                )

            self.events.send(
                LLMResponseCompletedEvent(
                    original=original_for_event,
                    text=final_text_for_event,
                    plugin_name="aws",
                )
            )

        except ClientError as e:
            error_code = (
                e.response.get("Error", {}).get("Code", "Unknown")
                if hasattr(e, "response")
                else "Unknown"
            )
            error_msg = (
                e.response.get("Error", {}).get("Message", str(e))
                if hasattr(e, "response")
                else str(e)
            )
            self.logger.error(
                f"AWS Bedrock API error: {error_code} - {error_msg}", exc_info=True
            )
            llm_response = LLMResponseEvent(None, error_msg, exception=e)
        except Exception as e:
            self.logger.error(
                f"Unexpected error in converse: {type(e).__name__}: {str(e)}",
                exc_info=True,
            )
            llm_response = LLMResponseEvent(
                None, f"Unexpected error: {str(e)}", exception=e
            )

        return llm_response

    async def converse_stream(self, *args, **kwargs) -> LLMResponseEvent[Any]:
        """
        Streaming version of converse using Bedrock's ConverseStream API.
        """
        client = await self.client

        if "modelId" not in kwargs:
            kwargs["modelId"] = self.model

        # Add tools if available
        tools = self.get_available_functions()
        if tools:
            converted_tools = self._convert_tools_to_provider_format(tools)
            kwargs["toolConfig"] = {"tools": converted_tools}

        # Ensure the AI remembers the past conversation
        new_messages = kwargs.get("messages", [])
        if hasattr(self, "_conversation") and self._conversation:
            old_messages = [m.original for m in self._conversation.messages]
            kwargs["messages"] = old_messages + new_messages
            normalized_messages = self._normalize_message(new_messages)
            for msg in normalized_messages:
                self._conversation.messages.append(msg)

        if self._instructions:
            kwargs["system"] = [{"text": self._instructions}]

        try:
            system_param = kwargs.get("system")

            # Helper to consume stream in a thread (stream iteration is blocking I/O)
            def _consume_stream():
                response = client.converse_stream(**kwargs)
                stream = response.get("stream")
                if not stream:
                    return None, [], []

                events = []
                for event in stream:
                    events.append(event)
                return stream, events, response

            try:
                stream, events, response = await asyncio.to_thread(_consume_stream)
            except ClientError as e:
                error_code = (
                    e.response.get("Error", {}).get("Code", "Unknown")
                    if hasattr(e, "response")
                    else "Unknown"
                )
                error_msg = (
                    e.response.get("Error", {}).get("Message", str(e))
                    if hasattr(e, "response")
                    else str(e)
                )
                self.logger.error(
                    f"AWS Bedrock API error in converse_stream: {error_code} - {error_msg}",
                    exc_info=True,
                )
                raise

            if not stream:
                self.logger.error("converse_stream response has no 'stream' field")
                llm_response = LLMResponseEvent(None, "No stream in response")
                return llm_response

            text_parts: List[str] = []
            accumulated_calls: List[NormalizedToolCallItem] = []
            last_event = None

            for event in events:
                last_event = event
                self._process_stream_event(event, text_parts, accumulated_calls)

            messages = kwargs["messages"][:]
            MAX_ROUNDS = 3
            rounds = 0
            seen: set[tuple[str, str, str]] = set()

            if accumulated_calls:
                assistant_content = []
                for tool_call in accumulated_calls:
                    assistant_content.append(
                        {
                            "toolUse": {
                                "toolUseId": tool_call["id"],
                                "name": tool_call["name"],
                                "input": tool_call["arguments_json"],
                            }
                        }
                    )
                assistant_msg_from_stream = {
                    "role": "assistant",
                    "content": assistant_content,
                }
                messages.append(assistant_msg_from_stream)

            while accumulated_calls and rounds < MAX_ROUNDS:
                triples, seen = await self._dedup_and_execute(
                    cast(List[NormalizedToolCallItem], accumulated_calls),
                    seen=seen,
                    max_concurrency=8,
                    timeout_s=30,
                )

                if not triples:
                    self.logger.warning("No tool execution results despite tool calls")
                    break

                tool_result_blocks = []
                for tc, res, err in triples:
                    if err:
                        self.logger.error(f"Tool {tc['name']} execution error: {err}")
                        tool_response = str(err)
                    else:
                        # Convert result to string format (AWS expects text, not json content type)
                        if isinstance(res, (dict, list)):
                            tool_response = json.dumps(res)
                        elif isinstance(res, str):
                            tool_response = res
                        else:
                            tool_response = str(res)

                    tool_result_blocks.append(
                        {
                            "toolUseId": tc["id"],
                            "content": [{"text": tool_response}],
                        }
                    )

                user_tool_results_msg = {
                    "role": "user",
                    "content": [{"toolResult": tr} for tr in tool_result_blocks],
                }
                messages = messages + [user_tool_results_msg]
                follow_up_kwargs = {
                    "modelId": self.model,
                    "messages": messages,
                    "toolConfig": kwargs.get("toolConfig", {}),
                }
                if system_param:
                    follow_up_kwargs["system"] = system_param

                follow_up_response = client.converse_stream(**follow_up_kwargs)

                accumulated_calls = []
                follow_up_text_parts: List[str] = []
                follow_up_stream = follow_up_response.get("stream")
                for event in follow_up_stream:
                    last_event = event
                    self._process_stream_event(
                        event, follow_up_text_parts, accumulated_calls
                    )

                if follow_up_text_parts:
                    text_parts.extend(follow_up_text_parts)

                if accumulated_calls:
                    follow_up_assistant_content = []
                    for tool_call in accumulated_calls:
                        follow_up_assistant_content.append(
                            {
                                "toolUse": {
                                    "toolUseId": tool_call["id"],
                                    "name": tool_call["name"],
                                    "input": tool_call["arguments_json"],
                                }
                            }
                        )
                    follow_up_assistant_msg = {
                        "role": "assistant",
                        "content": follow_up_assistant_content,
                    }
                    messages.append(follow_up_assistant_msg)

                if follow_up_text_parts and not accumulated_calls:
                    break

                rounds += 1

            if accumulated_calls:
                final_kwargs = {
                    "modelId": self.model,
                    "messages": messages,
                }
                if system_param:
                    final_kwargs["system"] = system_param

                def _consume_final_stream():
                    response = client.converse_stream(**final_kwargs)
                    stream = response.get("stream")
                    if not stream:
                        return []
                    events = []
                    for event in stream:
                        events.append(event)
                    return events

                final_events = await asyncio.to_thread(_consume_final_stream)
                final_text_parts: List[str] = []
                for event in final_events:
                    last_event = event
                    self._process_stream_event(
                        event, final_text_parts, accumulated_calls
                    )
                if final_text_parts:
                    text_parts.extend(final_text_parts)

            total_text = "".join(text_parts)
            llm_response = LLMResponseEvent(last_event, total_text)
            self.events.send(
                LLMResponseCompletedEvent(
                    original=last_event, text=total_text, plugin_name="aws"
                )
            )

        except ClientError as e:
            error_msg = f"AWS Bedrock streaming error: {str(e)}"
            llm_response = LLMResponseEvent(None, error_msg)

        return llm_response

    def _process_stream_event(
        self,
        event: Dict[str, Any],
        text_parts: List[str],
        accumulated_calls: List[NormalizedToolCallItem],
    ):
        """Process a streaming event from AWS."""
        # Forward the native event
        self.events.send(events.AWSStreamEvent(plugin_name="aws", event_data=event))

        # Handle content block delta (text)
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"]["delta"]
            if "text" in delta:
                text_parts.append(delta["text"])
                self.events.send(
                    LLMResponseChunkEvent(
                        plugin_name="aws",
                        content_index=event["contentBlockDelta"].get(
                            "contentBlockIndex", 0
                        ),
                        item_id="",
                        output_index=0,
                        sequence_number=0,
                        delta=delta["text"],
                    )
                )

        # Handle tool use
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
            delta = event["contentBlockDelta"]["delta"]
            if "toolUse" in delta:
                idx = event["contentBlockDelta"].get("contentBlockIndex", 0)
                if idx in self._pending_tool_uses_by_index:
                    input_data = delta["toolUse"].get("input", "")
                    self._pending_tool_uses_by_index[idx]["parts"].append(input_data)

        if "contentBlockStop" in event:
            idx = event["contentBlockStop"].get("contentBlockIndex", 0)
            pending = self._pending_tool_uses_by_index.pop(idx, None)
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
                accumulated_calls.append(tool_call_item)

    def _extract_text_from_response(self, response: Dict[str, Any]) -> str:
        """Extract text content from AWS response."""
        output = response.get("output", {})
        message = output.get("message", {})
        content = message.get("content", [])

        text_parts = []
        for item in content:
            if "text" in item:
                text_parts.append(item["text"])

        return "".join(text_parts)

    def _extract_tool_calls_from_response(
        self, response: Dict[str, Any]
    ) -> List[NormalizedToolCallItem]:
        """Extract tool calls from AWS response."""
        tool_calls: List[NormalizedToolCallItem] = []

        output = response.get("output", {})
        if not output:
            return tool_calls

        message = output.get("message", {})
        if not message:
            return tool_calls

        content = message.get("content", [])
        if not content:
            return tool_calls

        for item in content:
            if "toolUse" in item:
                tool_use = item["toolUse"]
                tool_call: NormalizedToolCallItem = {
                    "type": "tool_call",
                    "id": tool_use.get("toolUseId", ""),
                    "name": tool_use.get("name", ""),
                    "arguments_json": tool_use.get("input", {}),
                }
                tool_calls.append(tool_call)

        return tool_calls

    def _convert_tools_to_provider_format(
        self, tools: List[ToolSchema]
    ) -> List[Dict[str, Any]]:
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

    @staticmethod
    def _normalize_message(aws_messages: Any) -> List["Message"]:
        """Normalize AWS messages to internal Message format."""
        from vision_agents.core.agents.conversation import Message

        if isinstance(aws_messages, str):
            aws_messages = [{"content": [{"text": aws_messages}], "role": "user"}]

        if not isinstance(aws_messages, (List, tuple)):
            aws_messages = [aws_messages]

        messages: List[Message] = []
        for m in aws_messages:
            if isinstance(m, dict):
                content_items = m.get("content", [])
                # Extract text from content blocks
                text_parts = []
                for item in content_items:
                    if isinstance(item, dict) and "text" in item:
                        text_parts.append(item["text"])
                    elif isinstance(item, str):
                        text_parts.append(item)
                content = " ".join(text_parts)
                role = m.get("role", "user")
            else:
                content = str(m)
                role = "user"
            message = Message(original=m, content=content, role=role)
            messages.append(message)

        return messages
