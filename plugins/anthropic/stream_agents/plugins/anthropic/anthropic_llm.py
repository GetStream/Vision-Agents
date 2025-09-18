from typing import Optional, List, TYPE_CHECKING, Any, Dict, Tuple

import anthropic
from anthropic import AsyncAnthropic, AsyncStream
from anthropic.types import (
    RawMessageStreamEvent,
    Message as ClaudeMessage,
    RawContentBlockDeltaEvent,
    RawMessageStopEvent,
)

from stream_agents.core.llm.llm import LLM, LLMResponse
from stream_agents.core.llm.llm_types import ToolSchema, NormalizedToolCallItem

from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant

from stream_agents.core.llm.types import StandardizedTextDeltaEvent
from stream_agents.core.processors import BaseProcessor

if TYPE_CHECKING:
    from stream_agents.core.agents.conversation import Message


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

        from stream_agents.plugins import anthropic
        llm = anthropic.LLM(model="claude-opus-4-1-20250805")
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        client: Optional[AsyncAnthropic] = None,
    ):
        """
        Initialize the ClaudeLLM class.

        Args:
            model (str): The model to use. https://docs.anthropic.com/en/docs/about-claude/models/overview
            api_key: optional API key. by default loads from ANTHROPIC_API_KEY
            client: optional Anthropic client. by default creates a new client object.
        """
        super().__init__()
        self.model = model

        if client is not None:
            self.client = client
        else:
            self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def simple_response(
        self,
        text: str,
        processors: Optional[List[BaseProcessor]] = None,
        participant: Participant = None,
    ):
        """
        simple_response is a standardized way (across openai, claude, gemini etc.) to create a response.

        Args:
            text: The text to respond to
            processors: list of processors (which contain state) about the video/voice AI
            participant: optionally the participant object

        Examples:

            llm.simple_response("say hi to the user, be mean")
        """
        return await self.create_message(
            messages=[{"role": "user", "content": text}], max_tokens=1000
        )

    async def create_message(self, *args, **kwargs) -> LLMResponse:
        """
        create_message gives you full support/access to the native Claude message.create method
        this method wraps the Claude method and ensures we broadcast an event which the agent class hooks into
        """
        if "model" not in kwargs:
            kwargs["model"] = self.model

        if "stream" not in kwargs:
            kwargs["stream"] = True

        # Add tools if available - use Anthropic format
        tools = self.get_available_functions()
        if tools:
            kwargs["tools"] = self._convert_tools_to_provider_format(tools)
            kwargs.setdefault("tool_choice", {"type": "any"})

        # ensure the AI remembers the past conversation
        new_messages = kwargs["messages"]
        if hasattr(self, '_conversation') and self._conversation:
            old_messages = [m.original for m in self._conversation.messages]
            kwargs["messages"] = old_messages + new_messages
            # Add messages to conversation
            normalized_messages = self._normalize_message(new_messages)
            for msg in normalized_messages:
                self._conversation.messages.append(msg)

        self.emit("before_llm_response", self._normalize_message(kwargs["messages"]))

        original = await self.client.messages.create(*args, **kwargs)
        if isinstance(original, ClaudeMessage):
            # Extract text from Claude's response format - safely handle all text blocks
            text = self._concat_text_blocks(original.content)
            llm_response = LLMResponse(original, text)
            
            # Check if there were function calls in the response
            function_calls = self._extract_tool_calls_from_response(original)
            if function_calls:
                # Execute the functions and get results
                function_results = []
                for tool_call in function_calls:
                    try:
                        result = self.call_function(tool_call['name'], tool_call['arguments'])
                        function_results.append({
                            'name': tool_call['name'],
                            'result': result
                        })
                    except Exception as e:
                        function_results.append({
                            'name': tool_call['name'],
                            'result': {'error': str(e)}
                        })
                
                # Create function result messages for Claude
                function_messages = self._create_tool_result_message(function_calls, [r['result'] for r in function_results])
                
                # Send the function results back to Claude for a coherent response
                if function_messages:
                    # Include the assistant message that contained the tool_use blocks
                    assistant_msg = {"role": "assistant", "content": original.content}
                    tool_result_user_msg = function_messages[0]  # Single user message with tool_result blocks
                    
                    follow_up_messages = kwargs["messages"] + [assistant_msg, tool_result_user_msg]
                    
                    # Get a follow-up response from Claude
                    follow_up_response = await self.client.messages.create(
                        model=self.model,
                        messages=follow_up_messages,
                        max_tokens=1000,
                        # Don't include tools for follow-up - we want text response, not more tool calls
                        stream=False
                    )
                    
                    # Extract text from the follow-up response
                    follow_up_text = self._concat_text_blocks(follow_up_response.content)
                    llm_response = LLMResponse(follow_up_response, follow_up_text)
                            
        elif isinstance(original, AsyncStream):
            stream: AsyncStream[RawMessageStreamEvent] = original
            text_parts: List[str] = []
            accumulated_tool_calls: List[NormalizedToolCallItem] = []
            current_tool_call = None
            
            async for event in stream:
                llm_response_optional = self._standardize_and_emit_event(
                    event, text_parts
                )
                if llm_response_optional is not None:
                    llm_response = llm_response_optional
                
                # Check for tool calls in streaming chunks
                new_tool_calls, current_tool_call = self._extract_tool_calls_from_stream_chunk(event, current_tool_call)
                if new_tool_calls:
                    accumulated_tool_calls.extend(new_tool_calls)
            
            # If we have accumulated tool calls, execute them and get a follow-up response
            if accumulated_tool_calls:
                # Execute the functions and get results
                function_results = []
                for tool_call in accumulated_tool_calls:
                    try:
                        result = self.call_function(tool_call['name'], tool_call['arguments'])
                        function_results.append({
                            'name': tool_call['name'],
                            'result': result
                        })
                    except Exception as e:
                        function_results.append({
                            'name': tool_call['name'],
                            'result': {'error': str(e)}
                        })
                
                # Create function result messages for Claude
                function_messages = self._create_tool_result_message(accumulated_tool_calls, [r['result'] for r in function_results])
                
                # Send the function results back to Claude for a coherent response
                if function_messages:
                    # Include the assistant message that contained the tool_use blocks
                    # We need to reconstruct the assistant message from the accumulated tool calls
                    assistant_content = []
                    for tool_call in accumulated_tool_calls:
                        assistant_content.append({
                            "type": "tool_use",
                            "id": tool_call["id"],
                            "name": tool_call["name"],
                            "input": tool_call["arguments"]
                        })
                    
                    assistant_msg = {"role": "assistant", "content": assistant_content}
                    tool_result_user_msg = function_messages[0]  # Single user message with tool_result blocks
                    
                    follow_up_messages = kwargs["messages"] + [assistant_msg, tool_result_user_msg]
                    
                    # Get a follow-up response from Claude (streaming)
                    follow_up_stream = await self.client.messages.create(
                        model=self.model,
                        messages=follow_up_messages,
                        max_tokens=1000,
                        # Don't include tools for follow-up - we want text response, not more tool calls
                        stream=True
                    )
                    
                    # Process the follow-up stream
                    follow_up_text_parts = []
                    follow_up_chunk = None
                    
                    async for event in follow_up_stream:
                        follow_up_chunk = event
                        llm_response_optional = self._standardize_and_emit_event(
                            event, follow_up_text_parts
                        )
                        if llm_response_optional is not None:
                            llm_response = llm_response_optional
                    
                    # Use the follow-up response if we got one
                    if follow_up_text_parts:
                        total_text = "".join(follow_up_text_parts)
                        llm_response = LLMResponse(follow_up_chunk, total_text)

        self.emit("after_llm_response", llm_response)

        return llm_response

    def _standardize_and_emit_event(
        self, event: RawMessageStreamEvent, text_parts: List[str]
    ) -> Optional[LLMResponse]:
        """
        Forwards the events and also send out a standardized version (the agent class hooks into that)
        """
        # forward the native event
        self.emit("claude_event", event)

        # send a standardized version for delta and response
        if event.type == "content_block_delta":
            delta_event: RawContentBlockDeltaEvent = event
            if hasattr(delta_event.delta, "text") and delta_event.delta.text:
                text_parts.append(delta_event.delta.text)

                standardized_event = StandardizedTextDeltaEvent(
                    content_index=delta_event.index,
                    item_id="",
                    output_index=0,
                    sequence_number=0,
                    type="response.output_text.delta",
                    delta=delta_event.delta.text,
                )
                self.emit("standardized.output_text.delta", standardized_event)
        elif event.type == "message_stop":
            stop_event: RawMessageStopEvent = event
            total_text = "".join(text_parts)
            llm_response = LLMResponse(stop_event, total_text)
            return llm_response
        return None

    @staticmethod
    def _normalize_message(claude_messages: Any) -> List["Message"]:
        from stream_agents.core.agents.conversation import Message

        if isinstance(claude_messages, str):
            claude_messages = [
                {"content": claude_messages, "role": "user", "type": "text"}
            ]

        if not isinstance(claude_messages, (List, tuple)):
            claude_messages = [claude_messages]

        messages: List[Message] = []
        for m in claude_messages:
            if isinstance(m, dict):
                content = m.get("content", "")
                role = m.get("role", "user")
            else:
                content = str(m)
                role = "user"
            message = Message(original=m, content=content, role=role)
            messages.append(message)

        return messages

    def _convert_tools_to_provider_format(self, tools: List[ToolSchema]) -> List[Dict[str, Any]]:
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
                "input_schema": tool["parameters_schema"]
            }
            anthropic_tools.append(anthropic_tool)
        return anthropic_tools

    def _extract_tool_calls_from_response(self, response: Any) -> List[NormalizedToolCallItem]:
        """
        Extract tool calls from Anthropic response.
        
        Args:
            response: Anthropic response object
            
        Returns:
            List of normalized tool call items
        """
        tool_calls = []
        
        if hasattr(response, 'content') and response.content:
            for content_block in response.content:
                if hasattr(content_block, 'type') and content_block.type == "tool_use":
                    tool_calls.append({
                        "type": "tool_call",
                        "id": content_block.id,  # Critical: capture the id for tool_result
                        "name": content_block.name,
                        "arguments": content_block.input or {}  # dict, not arguments_json
                    })
        
        return tool_calls

    def _extract_tool_calls_from_stream_chunk(self, chunk: Any, current_tool_call: Optional[NormalizedToolCallItem] = None) -> tuple[List[NormalizedToolCallItem], Optional[NormalizedToolCallItem]]:
        """
        Extract tool calls from Anthropic streaming chunk.
        
        Args:
            chunk: Anthropic streaming event
            current_tool_call: Currently accumulating tool call (unused in this implementation)
            
        Returns:
            Tuple of (completed tool calls, current tool call being accumulated)
        """
        tool_calls = []
        
        if hasattr(chunk, 'type'):
            if chunk.type == "content_block_start":
                if hasattr(chunk, 'content_block') and hasattr(chunk.content_block, 'type'):
                    if chunk.content_block.type == "tool_use":
                        # Initialize pending tool use tracking
                        if not hasattr(self, '_pending_tool_uses'):
                            self._pending_tool_uses = {}
                        
                        cb = chunk.content_block
                        self._pending_tool_uses[cb.id] = {
                            "id": cb.id,
                            "name": cb.name,
                            "parts": []
                        }

            elif chunk.type == "content_block_delta":
                d = getattr(chunk, 'delta', None)
                if getattr(d, 'type', None) == "input_json_delta":
                    # Accumulate partial JSON strings
                    pj = getattr(d, 'partial_json', None)
                    if pj is not None and hasattr(self, '_pending_tool_uses') and self._pending_tool_uses:
                        # Find the most recent tool_use (in practice, there should be only one active)
                        last_tool = next(reversed(self._pending_tool_uses.values()))
                        last_tool["parts"].append(pj)

            elif chunk.type == "content_block_stop":
                # Finalize the tool use with accumulated JSON
                if hasattr(self, '_pending_tool_uses') and self._pending_tool_uses:
                    last_tool = next(reversed(self._pending_tool_uses.values()))
                    import json
                    buf = "".join(last_tool["parts"]).strip() or "{}"
                    try:
                        args = json.loads(buf)
                    except Exception as e:
                        args = {}
                    
                    tool_calls.append({
                        "type": "tool_call",
                        "id": last_tool["id"],
                        "name": last_tool["name"],
                        "arguments": args
                    })
                    
                    # Clean up
                    self._pending_tool_uses.pop(last_tool["id"], None)
        
        return tool_calls, None

    def _create_tool_result_message(self, tool_calls: List[NormalizedToolCallItem], results: List[Any]) -> List[Dict[str, Any]]:
        """
        Create tool result messages for Anthropic.
        
        Args:
            tool_calls: List of tool calls that were executed
            results: List of results from function execution
            
        Returns:
            List of tool result messages in Anthropic format
        """
        import json
        
        # Create a single user message with tool_result blocks
        blocks = []
        for tool_call, result in zip(tool_calls, results):
            # Convert result to string if it's not already
            if isinstance(result, (str, int, float)):
                payload = str(result)
            else:
                payload = json.dumps(result)
            
            blocks.append({
                "type": "tool_result",
                "tool_use_id": tool_call["id"],  # Critical: must match tool_use.id
                "content": payload
            })
        
        return [{"role": "user", "content": blocks}]

    def _concat_text_blocks(self, content):
        """Safely extract text from all text blocks in content."""
        out = []
        for b in content or []:
            if getattr(b, 'type', None) == "text" and getattr(b, 'text', None):
                out.append(b.text)
        return "".join(out)
