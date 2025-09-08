from typing import Optional, List, Iterable, TYPE_CHECKING, Dict, Any

import anthropic
from anthropic import AsyncAnthropic

from stream_agents.core.llm.llm import LLM, LLMResponse
from stream_agents.core.llm.llm_types import NormalizedResponse, NormalizedToolCallItem

from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant
from stream_agents.core.processors import BaseProcessor

if TYPE_CHECKING:
    from stream_agents.core.agents.conversation import Message


class ClaudeLLM(LLM):
    '''
    Manually keep history
    '''
    def __init__(self, model: str, api_key: Optional[str] = None, client: Optional[AsyncAnthropic] = None):
        super().__init__()
        self.model = model

        if client is not None:
            self.client = client
        else:
            self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def create_message(self, *args, **kwargs) -> LLMResponse:
        if "model" not in kwargs:
            kwargs["model"] = self.model

        # Add tools if functions are available
        available_functions = self.get_available_functions()
        if available_functions:
            kwargs["tools"] = [self._tool_schema_to_claude_tool(schema) for schema in available_functions]

        # ensure the AI remembers the past conversation
        new_messages =  kwargs["messages"]
        old_messages = [m.original for m in self._conversation.messages]
        kwargs["messages"] = old_messages + new_messages

        self._conversation.add_messages(self._normalize_message(new_messages))

        if hasattr(self, "before_response_listener"):
            self.before_response_listener(new_messages)

        original = await self.client.messages.create(*args, **kwargs)

        # Process tool calls if present
        normalized_response = self._normalize_claude_response(original)
        if normalized_response.get("output"):
            normalized_response = self.process_tool_calls(normalized_response)

        # Extract text from Claude's response format
        text = original.content[0].text if original.content else ""
        llm_response = LLMResponse(original, text)
        if hasattr(self, "after_response_listener"):
            await self.after_response_listener(llm_response)
        return llm_response

    async def simple_response(self, text: str, processors: Optional[List[BaseProcessor]] = None, participant: Participant = None):
        return await self.create_message(
            messages=[{"role": "user", "content": text}],
            max_tokens=1000
        )

    @staticmethod
    def _normalize_message(claude_messages: Iterable["Message"]) -> List["Message"]:
        from stream_agents.core.agents.conversation import Message
        
        if isinstance(claude_messages, str):
            claude_messages = [{"content": claude_messages, "role": "user", "type": "text"}]

        if not isinstance(claude_messages, List):
            claude_messages = [claude_messages]

        messages : List[Message] = []
        for m in claude_messages:
            message = Message(original=m, content=m["content"], role=m["role"])
            messages.append(message)

        return messages

    def _tool_schema_to_claude_tool(self, schema) -> Dict[str, Any]:
        """Convert a tool schema to Claude tool format."""
        return {
            "name": schema["name"],
            "description": schema.get("description", ""),
            "input_schema": schema["parameters_schema"]
        }

    def _normalize_claude_response(self, response) -> NormalizedResponse:
        """Convert Claude response to normalized format."""
        output = []
        
        # Handle text content
        if response.content:
            for content_block in response.content:
                if content_block.type == "text":
                    output.append({
                        "type": "text",
                        "text": content_block.text
                    })
                elif content_block.type == "tool_use":
                    tool_call_item: NormalizedToolCallItem = {
                        "type": "tool_call",
                        "name": content_block.name,
                        "arguments_json": content_block.input
                    }
                    output.append(tool_call_item)
        
        return {
            "id": response.id,
            "model": self.model,
            "status": "completed",
            "output": output,
            "output_text": response.content[0].text if response.content and response.content[0].type == "text" else "",
            "raw": response
        }