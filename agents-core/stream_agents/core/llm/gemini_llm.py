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

        # Check if we have functions available
        available_functions = self.get_available_functions()
        if available_functions:
            # Use generate_content directly for function calling
            tools = [self._tool_schema_to_gemini_tool(schema) for schema in available_functions]
            config = genai.types.GenerateContentConfig(tools=tools)
            response = self.client.models.generate_content(
                model=f"models/{self.model}",
                contents=kwargs.get("message", ""),
                config=config
            )
        else:
            # Use chat interface for regular messages
            if self.chat is None:
                self.chat = self.client.chats.create(model=self.model)
            response = self.chat.send_message(*args, **kwargs)

        # Process tool calls if present
        normalized_response = self._normalize_gemini_response(response)
        if normalized_response.get("output"):
            normalized_response = self.process_tool_calls(normalized_response)

        # Extract text from processed response or original response
        text = normalized_response.get("output_text", response.text if response.text else "")
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

    def _tool_schema_to_gemini_tool(self, schema) -> genai.types.Tool:
        """Convert a tool schema to Gemini tool format."""
        function_declaration = genai.types.FunctionDeclaration(
            name=schema["name"],
            description=schema.get("description", ""),
            parameters=schema["parameters_schema"]
        )
        return genai.types.Tool(
            function_declarations=[function_declaration]
        )

    def _normalize_gemini_response(self, response) -> NormalizedResponse:
        """Convert Gemini response to normalized format."""
        output = []
        
        # Handle tool calls if present
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                for part in candidate.content.parts:
                    if hasattr(part, 'function_call') and part.function_call is not None:
                        tool_call_item: NormalizedToolCallItem = {
                            "type": "tool_call",
                            "name": part.function_call.name,
                            "arguments_json": part.function_call.args
                        }
                        output.append(tool_call_item)
                    elif hasattr(part, 'text') and part.text:
                        output.append({
                            "type": "text",
                            "text": part.text
                        })
        
        # Handle text content if no parts were processed
        if not output and response.text:
            output.append({
                "type": "text",
                "text": response.text
            })
        
        return {
            "id": getattr(response, 'id', ''),
            "model": self.model,
            "status": "completed",
            "output": output,
            "output_text": response.text or "",
            "raw": response
        }
    
    def _generate_conversational_response(self, tool_results: list, original_response: NormalizedResponse) -> str:
        """Generate a conversational response based on tool results using Gemini."""
        try:
            # Create a simple prompt to generate a conversational response
            results_text = []
            for result in tool_results:
                result_data = result["result_json"]
                function_name = result["name"]
                if isinstance(result_data, dict):
                    if "result" in result_data:
                        results_text.append(f"{function_name} result: {result_data['result']}")
                    else:
                        results_text.append(f"{function_name} result: {result_data}")
                else:
                    results_text.append(f"{function_name} result: {result_data}")
            
            prompt = f"""The user asked a question and I have the following results from my functions:
{chr(10).join(results_text)}

Please provide a natural, conversational response based on these results. Be helpful and informative."""
            
            # Generate a conversational response using the chat interface
            if self.chat is None:
                self.chat = self.client.chats.create(model=self.model)
            
            follow_up_response = self.chat.send_message(prompt)
            return follow_up_response.text
            
        except Exception as e:
            # If there's an error, return None to use fallback formatting
            print(f"Error generating conversational response: {e}")
            return None