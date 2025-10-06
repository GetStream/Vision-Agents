from typing import Optional, List, TYPE_CHECKING, Any, Dict
import json
import anthropic
from anthropic import AsyncAnthropic, AsyncStream
from anthropic.types import (
    RawMessageStreamEvent,
    Message as ClaudeMessage,
    RawContentBlockDeltaEvent,
    RawMessageStopEvent,
)

from vision_agents.core.llm.llm import LLM, LLMResponseEvent
from vision_agents.core.llm.llm_types import ToolSchema, NormalizedToolCallItem

from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant

from vision_agents.core.llm.events import LLMResponseChunkEvent, LLMResponseCompletedEvent
from vision_agents.core.processors import Processor
from . import events

if TYPE_CHECKING:
    from vision_agents.core.agents.conversation import Message


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
        self.events.register_events_from_module(events)
        self.model = model
        self._pending_tool_uses_by_index: Dict[int, Dict[str, Any]] = {}  # index -> {id, name, parts: []}

        if client is not None:
            self.client = client
        else:
            self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def simple_response(
        self,
        text: str,
        processors: Optional[List[Processor]] = None,
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

    async def create_message(self, *args, **kwargs) -> LLMResponseEvent[Any]:
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
            kwargs.setdefault("tool_choice", {"type": "auto"})

        # ensure the AI remembers the past conversation
        new_messages = kwargs["messages"]
        if hasattr(self, '_conversation') and self._conversation:
            old_messages = [m.original for m in self._conversation.messages]
            kwargs["messages"] = old_messages + new_messages
            # Add messages to conversation
            normalized_messages = self._normalize_message(new_messages)
            for msg in normalized_messages:
                self._conversation.messages.append(msg)

        # Note: Message history is tracked in _conversation, no need to emit as event here

        original = await self.client.messages.create(*args, **kwargs)
        if isinstance(original, ClaudeMessage):
            # Extract text from Claude's response format - safely handle all text blocks
            text = self._concat_text_blocks(original.content)
            llm_response = LLMResponseEvent(original, text)
            
            # Multi-hop tool calling loop for non-streaming
            function_calls = self._extract_tool_calls_from_response(original)
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
                    
                    # Build tool_result user message
                    assistant_content = []
                    tool_result_blocks = []
                    for tc, res, err in triples:
                        assistant_content.append({
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["name"],
                            "input": tc["arguments_json"],
                        })
                        
                        payload = self._sanitize_tool_output(res)
                        tool_result_blocks.append({
                            "type": "tool_result",
                            "tool_use_id": tc["id"],
                            "content": payload,
                        })

                    assistant_msg = {"role": "assistant", "content": assistant_content}
                    user_tool_results_msg = {"role": "user", "content": tool_result_blocks}
                    messages = messages + [assistant_msg, user_tool_results_msg]

                    # Ask again WITH tools so Claude can do another hop
                    tools_cfg = {
                        "tools": self._convert_tools_to_provider_format(self.get_available_functions()),
                        "tool_choice": {"type": "auto"},
                        "stream": False,
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": 1000,
                    }

                    follow_up_response = await self.client.messages.create(**tools_cfg)
                    
                    # Extract new tool calls from follow-up response
                    current_calls = self._extract_tool_calls_from_response(follow_up_response)
                    llm_response = LLMResponseEvent(follow_up_response, self._concat_text_blocks(follow_up_response.content))
                    rounds += 1
                
                # Finalization pass: no tools so Claude must answer in text
                if current_calls or rounds > 0:  # Only if we had tool calls
                    final_response = await self.client.messages.create(
                        model=self.model,
                        messages=messages,   # includes assistant tool_use + user tool_result blocks
                        stream=False,
                        max_tokens=1000
                    )
                    llm_response = LLMResponseEvent(final_response, self._concat_text_blocks(final_response.content))
                            
        elif isinstance(original, AsyncStream):
            stream: AsyncStream[RawMessageStreamEvent] = original
            text_parts: List[str] = []
            accumulated_calls: List[NormalizedToolCallItem] = []

            # 1) First round: read stream, gather initial tool_use calls
            async for event in stream:
                llm_response_optional = self._standardize_and_emit_event(event, text_parts)
                if llm_response_optional is not None:
                    llm_response = llm_response_optional
                # Collect tool_use calls as they complete (your helper already reconstructs args)
                new_calls, _ = self._extract_tool_calls_from_stream_chunk(event, None)
                if new_calls:
                    accumulated_calls.extend(new_calls)

            # Track full message history to reuse across rounds
            messages = kwargs["messages"][:]  # start from prior history
            MAX_ROUNDS = 3
            rounds = 0
            seen = set()

            # 2) While there are tool calls, execute -> return tool_result -> ask again (with tools)
            last_followup_stream = None
            while accumulated_calls and rounds < MAX_ROUNDS:
                # Execute calls concurrently with dedup
                triples, seen = await self._dedup_and_execute(accumulated_calls, seen=seen, max_concurrency=8, timeout_s=30)  # type: ignore[arg-type]

                # Build tool_result user message
                # Also reconstruct the assistant tool_use message that triggered these calls
                assistant_content = []
                executed_calls: List[NormalizedToolCallItem] = []
                for tc, res, err in triples:
                    executed_calls.append(tc)
                    assistant_content.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["arguments_json"],
                    })

                # tool_result blocks (sanitize to keep payloads safe)
                tool_result_blocks = []
                for tc, res, err in triples:
                    payload = self._sanitize_tool_output(res)
                    tool_result_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": payload,
                    })

                assistant_msg = {"role": "assistant", "content": assistant_content}
                user_tool_results_msg = {"role": "user", "content": tool_result_blocks}
                messages = messages + [assistant_msg, user_tool_results_msg]

                # Ask again WITH tools so Claude can do another hop
                tools_cfg = {
                    "tools": self._convert_tools_to_provider_format(self.get_available_functions()),
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
                    llm_response_optional = self._standardize_and_emit_event(ev, follow_up_text_parts)
                    if llm_response_optional is not None:
                        llm_response = llm_response_optional
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
                    messages=messages,   # includes assistant tool_use + user tool_result blocks
                    stream=True,
                    max_tokens=1000
                )
                final_text_parts: List[str] = []
                async for ev in final_stream:
                    last_followup_stream = ev
                    llm_response_optional = self._standardize_and_emit_event(ev, final_text_parts)
                    if llm_response_optional is not None:
                        llm_response = llm_response_optional
                if final_text_parts:
                    text_parts.append("".join(final_text_parts))

            # 4) Done -> return all collected text
            total_text = "".join(text_parts)
            llm_response = LLMResponseEvent(last_followup_stream or original, total_text) # type: ignore
            self.events.send(LLMResponseCompletedEvent(original=last_followup_stream or original, text=total_text, plugin_name="anthropic"))

        return llm_response

    def _standardize_and_emit_event(
        self, event: RawMessageStreamEvent, text_parts: List[str]
    ) -> Optional[LLMResponseEvent[Any]]:
        """
        Forwards the events and also send out a standardized version (the agent class hooks into that)
        """
        # forward the native event
        self.events.send(events.ClaudeStreamEvent(
            plugin_name="anthropic",
            event_data=event
        ))

        # send a standardized version for delta and response
        if event.type == "content_block_delta":
            delta_event: RawContentBlockDeltaEvent = event
            if hasattr(delta_event.delta, "text") and delta_event.delta.text:
                text_parts.append(delta_event.delta.text)

                self.events.send(LLMResponseChunkEvent(
                    plugin_name="antrhopic",
                    content_index=delta_event.index,
                    item_id="",
                    output_index=0,
                    sequence_number=0,
                    delta=delta_event.delta.text,
                ))
        elif event.type == "message_stop":
            stop_event: RawMessageStopEvent = event
            total_text = "".join(text_parts)
            llm_response = LLMResponseEvent(stop_event, total_text)
            return llm_response
        return None

    @staticmethod
    def _normalize_message(claude_messages: Any) -> List["Message"]:
        from vision_agents.core.agents.conversation import Message

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
                    tool_call: NormalizedToolCallItem = {
                        "type": "tool_call",
                        "id": content_block.id,  # Critical: capture the id for tool_result
                        "name": content_block.name,
                        "arguments_json": content_block.input or {}  # normalize to arguments_json
                    }
                    tool_calls.append(tool_call)
        
        return tool_calls

    def _extract_tool_calls_from_stream_chunk(self, chunk: Any, current_tool_call: Optional[NormalizedToolCallItem] = None) -> tuple[List[NormalizedToolCallItem], Optional[NormalizedToolCallItem]]:  # type: ignore[override]
        """
        Extract tool calls from Anthropic streaming chunk using index-keyed accumulation.
        Args:
            chunk: Anthropic streaming event
            current_tool_call: Currently accumulating tool call (unused in this implementation)
        Returns:
            Tuple of (completed tool calls, current tool call being accumulated)
        """
        tool_calls = []
        t = getattr(chunk, 'type', None)

        if t == "content_block_start":
            cb = getattr(chunk, 'content_block', None)
            if getattr(cb, 'type', None) == "tool_use":
                if cb is not None:
                    self._pending_tool_uses_by_index[chunk.index] = {
                        "id": cb.id,
                        "name": cb.name,
                        "parts": []
                    }

        elif t == "content_block_delta":
            d = getattr(chunk, 'delta', None)
            if getattr(d, 'type', None) == "input_json_delta":
                pj = getattr(d, 'partial_json', None)
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
                    "arguments_json": args
                }
                tool_calls.append(tool_call_item)
        return tool_calls, None

    def _create_tool_result_message(self, tool_calls: List[NormalizedToolCallItem], results: List[Any]) -> List[Dict[str, Any]]:
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
