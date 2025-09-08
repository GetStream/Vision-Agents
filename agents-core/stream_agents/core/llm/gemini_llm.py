from typing import Optional, List, TYPE_CHECKING, Dict, Any

from google import genai

from stream_agents.core.llm.llm import LLM, LLMResponse
from stream_agents.core.llm.llm_types import NormalizedResponse, NormalizedToolCallItem

from stream_agents.core.processors import BaseProcessor

if TYPE_CHECKING:
    from stream_agents.core.agents.conversation import Message


class GeminiLLM(LLM):
    '''
    Use the SDK to keep history. (which is partially manual)
    '''
    def __init__(self, model: str, api_key: Optional[str] = None, client: Optional[genai.Client] = None):
        super().__init__()
        self.model = model
        self.chat = None

        if client is not None:
            self.client = client
        else:
            self.client = genai.Client(api_key=api_key)

    async def simple_response(self, text: str, processors: Optional[List[BaseProcessor]] = None):
        return await self.send_message(
            message=text
        )

    # basically wrap the Gemini native endpoint
    async def send_message(self, *args, **kwargs):
        #if "model" not in kwargs:
        #    kwargs["model"] = self.model

        # Add tools if functions are available
        available_functions = self.get_available_functions()
        if available_functions:
            kwargs["tools"] = [self._tool_schema_to_gemini_tool(schema) for schema in available_functions]

        # initialize chat if needed
        if self.chat is None:
            self.chat = self.client.chats.create(model=self.model)

        # Generate content using the client
        response = self.chat.send_message(*args, **kwargs)

        # Process tool calls if present
        normalized_response = self._normalize_gemini_response(response)
        if normalized_response.get("output"):
            normalized_response = self.process_tool_calls(normalized_response)

        # Extract text from Gemini's response format
        text = response.text if response.text else ""
        return LLMResponse(response, text)

    @staticmethod
    def _normalize_message(gemini_input) -> List["Message"]:
        from stream_agents.core.agents.conversation import Message
        
        # standardize on input
        if isinstance(gemini_input, str):
            gemini_input = [
                gemini_input
            ]

        if not isinstance(gemini_input, List):
            gemini_input = [gemini_input]

        messages = []
        for i in gemini_input:
            message = Message(original=i, content=i)
            messages.append(message)

        return messages

    def _tool_schema_to_gemini_tool(self, schema) -> Dict[str, Any]:
        """Convert a tool schema to Gemini tool format."""
        return {
            "name": schema["name"],
            "description": schema.get("description", ""),
            "parameters": schema["parameters_schema"]
        }

    def _normalize_gemini_response(self, response) -> NormalizedResponse:
        """Convert Gemini response to normalized format."""
        output = []
        
        # Handle text content
        if response.text:
            output.append({
                "type": "text",
                "text": response.text
            })
        
        # Handle tool calls if present
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                for part in candidate.content.parts:
                    if hasattr(part, 'function_call'):
                        tool_call_item: NormalizedToolCallItem = {
                            "type": "tool_call",
                            "name": part.function_call.name,
                            "arguments_json": part.function_call.args
                        }
                        output.append(tool_call_item)
        
        return {
            "id": getattr(response, 'id', ''),
            "model": self.model,
            "status": "completed",
            "output": output,
            "output_text": response.text or "",
            "raw": response
        }