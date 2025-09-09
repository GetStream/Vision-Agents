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
    
    def _generate_conversational_response(self, tool_results: list, original_response: NormalizedResponse) -> str:
        """Generate a conversational response based on tool results using Claude."""
        try:
            import json
            
            # Prepare the conversation with tool results
            messages = []
            
            # Add the original user message if available
            if hasattr(self, '_conversation') and self._conversation:
                for message in reversed(self._conversation.messages):
                    if message.role == "user":
                        messages.append({
                            "role": "user",
                            "content": message.content
                        })
                        break
            
            # Add the assistant's tool calls
            tool_uses = []
            for i, result in enumerate(tool_results):
                tool_uses.append({
                    "id": f"call_{i}",
                    "name": result["name"],
                    "input": {}  # We don't need the original arguments
                })
            
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_use": tool_uses
            })
            
            # Add the tool results
            for i, result in enumerate(tool_results):
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": f"call_{i}",
                            "content": json.dumps(result["result_json"])
                        }
                    ]
                })
            
            # Add a system message to guide the response
            messages.insert(0, {
                "role": "user",
                "content": "You are a helpful assistant. The user asked you to perform some functions, and you have the results. Please provide a natural, conversational response based on these results. Be helpful and informative."
            })
            
            # Get a follow-up response from Claude
            follow_up_response = self.client.messages.create(
                model=self.model,
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            return follow_up_response.content[0].text
            
        except Exception as e:
            # If there's an error, return None to use fallback formatting
            print(f"Error generating conversational response: {e}")
            return None