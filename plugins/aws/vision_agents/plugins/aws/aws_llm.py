import os
from typing import Optional, List, TYPE_CHECKING, Any, Dict
import json
import boto3
from botocore.exceptions import ClientError

from vision_agents.core.llm.llm import LLM, LLMResponseEvent
from vision_agents.core.llm.llm_types import ToolSchema, NormalizedToolCallItem


from vision_agents.core.llm.events import LLMResponseChunkEvent, LLMResponseCompletedEvent
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
            
        self.client = boto3.client("bedrock-runtime", **session_kwargs)

        self.region_name = region_name

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
            kwargs["toolConfig"] = {
                "tools": self._convert_tools_to_provider_format(tools)
            }

        # Combine original instructions with markdown file contents
        enhanced_instructions = self._build_enhanced_instructions()
        if enhanced_instructions:
            kwargs["system"] = [{"text": enhanced_instructions}]

        # Ensure the AI remembers the past conversation
        new_messages = kwargs.get("messages", [])
        if hasattr(self, '_conversation') and self._conversation:
            old_messages = [m.original for m in self._conversation.messages]
            kwargs["messages"] = old_messages + new_messages
            # Add messages to conversation
            normalized_messages = self._normalize_message(new_messages)
            for msg in normalized_messages:
                self._conversation.messages.append(msg)

        try:
            response = self.client.converse(**kwargs)
            
            # Extract text from response
            text = self._extract_text_from_response(response)
            llm_response = LLMResponseEvent(response, text)
            
            # Handle tool calls if present
            function_calls = self._extract_tool_calls_from_response(response)
            if function_calls:
                messages = kwargs["messages"][:]
                MAX_ROUNDS = 3
                rounds = 0
                seen: set[tuple[str, str, str]] = set()
                current_calls = function_calls
                
                while current_calls and rounds < MAX_ROUNDS:
                    # Execute calls concurrently with dedup
                    triples, seen = await self._dedup_and_execute(current_calls, seen=seen, max_concurrency=8, timeout_s=30)  # type: ignore[arg-type]
                    
                    if not triples:
                        break
                    
                    # Build tool result message
                    tool_result_blocks = []
                    for tc, res, err in triples:
                        payload = self._sanitize_tool_output(res)
                        tool_result_blocks.append({
                            "toolUseId": tc["id"],
                            "content": [{"json": payload if isinstance(payload, dict) else {"result": payload}}],
                        })

                    # Add assistant message with tool use and user message with tool results
                    assistant_msg = {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": tc["id"],
                                    "name": tc["name"],
                                    "input": tc["arguments_json"],
                                }
                            }
                            for tc, _, _ in triples
                        ]
                    }
                    user_tool_results_msg = {
                        "role": "user",
                        "content": [{"toolResult": tr} for tr in tool_result_blocks]
                    }
                    messages = messages + [assistant_msg, user_tool_results_msg]

                    # Ask again WITH tools
                    follow_up_response = self.client.converse(
                        modelId=self.model,
                        messages=messages,
                        toolConfig=kwargs.get("toolConfig", {}),
                    )
                    
                    # Extract new tool calls
                    current_calls = self._extract_tool_calls_from_response(follow_up_response)
                    llm_response = LLMResponseEvent(follow_up_response, self._extract_text_from_response(follow_up_response))
                    rounds += 1
                
                # Final pass without tools
                if current_calls or rounds > 0:
                    final_response = self.client.converse(
                        modelId=self.model,
                        messages=messages,
                    )
                    llm_response = LLMResponseEvent(final_response, self._extract_text_from_response(final_response))
            
            self.events.send(LLMResponseCompletedEvent(original=response, text=text, plugin_name="aws"))

        except ClientError as e:
            error_msg = f"AWS Bedrock API error: {str(e)}"
            llm_response = LLMResponseEvent(None, error_msg, exception = e)
            
        return llm_response

    async def converse_stream(self, *args, **kwargs) -> LLMResponseEvent[Any]:
        """
        Streaming version of converse using Bedrock's ConverseStream API.
        """
        if "modelId" not in kwargs:
            kwargs["modelId"] = self.model

        # Add tools if available
        tools = self.get_available_functions()
        if tools:
            kwargs["toolConfig"] = {
                "tools": self._convert_tools_to_provider_format(tools)
            }

        # Ensure the AI remembers the past conversation
        new_messages = kwargs.get("messages", [])
        if hasattr(self, '_conversation') and self._conversation:
            old_messages = [m.original for m in self._conversation.messages]
            kwargs["messages"] = old_messages + new_messages
            normalized_messages = self._normalize_message(new_messages)
            for msg in normalized_messages:
                self._conversation.messages.append(msg)

        # Combine original instructions with markdown file contents
        enhanced_instructions = self._build_enhanced_instructions()
        if enhanced_instructions:
            kwargs["system"] = [{"text": enhanced_instructions}]

        try:
            response = self.client.converse_stream(**kwargs)
            stream = response.get('stream')
            
            text_parts: List[str] = []
            accumulated_calls: List[NormalizedToolCallItem] = []
            last_event = None
            
            # Process stream
            for event in stream:
                last_event = event
                self._process_stream_event(event, text_parts, accumulated_calls)
            
            # Handle multi-hop tool calling
            messages = kwargs["messages"][:]
            MAX_ROUNDS = 3
            rounds = 0
            seen: set[tuple[str, str, str]] = set()
            
            while accumulated_calls and rounds < MAX_ROUNDS:
                triples, seen = await self._dedup_and_execute(accumulated_calls, seen=seen, max_concurrency=8, timeout_s=30)  # type: ignore[arg-type]
                
                if not triples:
                    break
                
                # Build tool result messages
                tool_result_blocks = []
                for tc, res, err in triples:
                    payload = self._sanitize_tool_output(res)
                    tool_result_blocks.append({
                        "toolUseId": tc["id"],
                        "content": [{"json": payload if isinstance(payload, dict) else {"result": payload}}],
                    })

                assistant_msg = {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": tc["id"],
                                "name": tc["name"],
                                "input": tc["arguments_json"],
                            }
                        }
                        for tc, _, _ in triples
                    ]
                }
                user_tool_results_msg = {
                    "role": "user",
                    "content": [{"toolResult": tr} for tr in tool_result_blocks]
                }
                messages = messages + [assistant_msg, user_tool_results_msg]

                # Next round with tools
                follow_up_response = self.client.converse_stream(
                    modelId=self.model,
                    messages=messages,
                    toolConfig=kwargs.get("toolConfig", {}),
                )
                
                accumulated_calls = []
                follow_up_stream = follow_up_response.get('stream')
                for event in follow_up_stream:
                    last_event = event
                    self._process_stream_event(event, text_parts, accumulated_calls)
                
                rounds += 1
            
            # Final pass without tools
            if accumulated_calls or rounds > 0:
                final_response = self.client.converse_stream(
                    modelId=self.model,
                    messages=messages,
                )
                final_stream = final_response.get('stream')
                for event in final_stream:
                    last_event = event
                    self._process_stream_event(event, text_parts, accumulated_calls)
            
            total_text = "".join(text_parts)
            llm_response = LLMResponseEvent(last_event, total_text)
            self.events.send(LLMResponseCompletedEvent(original=last_event, text=total_text, plugin_name="aws"))
            
        except ClientError as e:
            error_msg = f"AWS Bedrock streaming error: {str(e)}"
            llm_response = LLMResponseEvent(None, error_msg)
            
        return llm_response

    def _process_stream_event(
        self, 
        event: Dict[str, Any], 
        text_parts: List[str],
        accumulated_calls: List[NormalizedToolCallItem]
    ):
        """Process a streaming event from AWS."""
        # Forward the native event
        self.events.send(events.AWSStreamEvent(
            plugin_name="aws",
            event_data=event
        ))
        
        # Handle content block delta (text)
        if 'contentBlockDelta' in event:
            delta = event['contentBlockDelta']['delta']
            if 'text' in delta:
                text_parts.append(delta['text'])
                self.events.send(LLMResponseChunkEvent(
                    plugin_name="aws",
                    content_index=event['contentBlockDelta'].get('contentBlockIndex', 0),
                    item_id="",
                    output_index=0,
                    sequence_number=0,
                    delta=delta['text'],
                ))
        
        # Handle tool use
        if 'contentBlockStart' in event:
            start = event['contentBlockStart'].get('start', {})
            if 'toolUse' in start:
                tool_use = start['toolUse']
                idx = event['contentBlockStart'].get('contentBlockIndex', 0)
                self._pending_tool_uses_by_index[idx] = {
                    "id": tool_use.get('toolUseId', ''),
                    "name": tool_use.get('name', ''),
                    "parts": []
                }
        
        if 'contentBlockDelta' in event:
            delta = event['contentBlockDelta']['delta']
            if 'toolUse' in delta:
                idx = event['contentBlockDelta'].get('contentBlockIndex', 0)
                if idx in self._pending_tool_uses_by_index:
                    input_data = delta['toolUse'].get('input', '')
                    self._pending_tool_uses_by_index[idx]['parts'].append(input_data)
        
        if 'contentBlockStop' in event:
            idx = event['contentBlockStop'].get('contentBlockIndex', 0)
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
                    "arguments_json": args
                }
                accumulated_calls.append(tool_call_item)

    def _extract_text_from_response(self, response: Dict[str, Any]) -> str:
        """Extract text content from AWS response."""
        output = response.get('output', {})
        message = output.get('message', {})
        content = message.get('content', [])
        
        text_parts = []
        for item in content:
            if 'text' in item:
                text_parts.append(item['text'])
        
        return "".join(text_parts)

    def _extract_tool_calls_from_response(self, response: Dict[str, Any]) -> List[NormalizedToolCallItem]:
        """Extract tool calls from AWS response."""
        tool_calls = []
        
        output = response.get('output', {})
        message = output.get('message', {})
        content = message.get('content', [])
        
        for item in content:
            if 'toolUse' in item:
                tool_use = item['toolUse']
                tool_call: NormalizedToolCallItem = {
                    "type": "tool_call",
                    "id": tool_use.get('toolUseId', ''),
                    "name": tool_use.get('name', ''),
                    "arguments_json": tool_use.get('input', {})
                }
                tool_calls.append(tool_call)
        
        return tool_calls

    def _convert_tools_to_provider_format(self, tools: List[ToolSchema]) -> List[Dict[str, Any]]:
        """
        Convert ToolSchema objects to AWS Bedrock format.
        
        Args:
            tools: List of ToolSchema objects
            
        Returns:
            List of tools in AWS Bedrock format
        """
        aws_tools = []
        for tool in tools:
            aws_tool = {
                "toolSpec": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "inputSchema": {
                        "json": tool["parameters_schema"]
                    }
                }
            }
            aws_tools.append(aws_tool)
        return aws_tools

    @staticmethod
    def _normalize_message(aws_messages: Any) -> List["Message"]:
        """Normalize AWS messages to internal Message format."""
        from vision_agents.core.agents.conversation import Message

        if isinstance(aws_messages, str):
            aws_messages = [
                {"content": [{"text": aws_messages}], "role": "user"}
            ]

        if not isinstance(aws_messages, (List, tuple)):
            aws_messages = [aws_messages]

        messages: List[Message] = []
        for m in aws_messages:
            if isinstance(m, dict):
                content_items = m.get("content", [])
                # Extract text from content blocks
                text_parts = []
                for item in content_items:
                    if isinstance(item, dict) and 'text' in item:
                        text_parts.append(item['text'])
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

