from typing import Optional, List, TypeVar, Callable, TYPE_CHECKING, Dict, Any, no_type_check
import json

from openai import OpenAI

from getstream.models import Response
import inspect
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant
from stream_agents.core.llm.llm import LLM, LLMResponse
from stream_agents.core.llm.llm_types import NormalizedResponse, NormalizedToolCallItem

from stream_agents.core.processors import BaseProcessor

if TYPE_CHECKING:
    from stream_agents.core.agents.conversation import Message


R = TypeVar("R")


def use_create(fn: Callable[..., R]) -> Callable[..., R]:
    return fn


class OpenAILLM(LLM):
    """
    The goal is to expose the regular/native openAI SDK methods,
    and only standardize the minimal feature set that's needed for the agent integration.
    """

    def __init__(
        self, model: str, api_key: Optional[str] = None, client: Optional[OpenAI] = None
    ):
        super().__init__()
        self.model = model
        self.openai_conversation: Optional[Any] = None
        self.conversation = None

        if client is not None:
            self.client: OpenAI = client
        elif api_key is not None and api_key != "":
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI()

    @no_type_check
    async def create_response(self, *args: Any, **kwargs: Any) -> LLMResponse[Response]:
        if "model" not in kwargs:
            kwargs["model"] = self.model

        # Check if we have functions available for function calling
        available_functions = self.get_available_functions()
        
        if available_functions:
            # Use function calling approach
            messages = self._prepare_messages(kwargs.get("input", ""))
            tools = [self._tool_schema_to_openai_tool(schema) for schema in available_functions]
        else:
            # Use conversations API for regular chat
            if self.openai_conversation is None:
                self.openai_conversation = self.client.conversations.create()
            kwargs["conversation"] = getattr(self.openai_conversation, "id", None)

        if hasattr(self, "before_response_listener") and "input" in kwargs:
            self.before_response_listener(self._normalize_message(kwargs["input"]))

        if available_functions:
            # Use chat completions API for function calling
            chat_kwargs = {
                "model": kwargs["model"],
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto"
            }

            response = self.client.chat.completions.create(**chat_kwargs)

            # Process tool calls if present
            normalized_response = self._normalize_openai_response(response)
            if normalized_response.get("output"):
                normalized_response = await self.process_tool_calls(normalized_response)
            # Use the processed response text
            response_text = normalized_response.get("output_text", response.choices[0].message.content or "")
            llm_response = LLMResponse[Response](response, response_text)
        else:
            # Use conversations API for regular chat
            conversation_response: Response = self.client.responses.create(*args, **kwargs)
            llm_response = LLMResponse[Response](
                conversation_response, conversation_response.output_text
            )
        if hasattr(self, "after_response_listener"):
            maybe_awaitable = self.after_response_listener(llm_response)
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable
        return llm_response

    async def simple_response(
        self,
        text: str,
        processors: Optional[List[BaseProcessor]] = None,
        participant: Participant = None,
    ) -> LLMResponse[Response]:
        instructions = None
        if self.conversation is not None:
            instructions = self.conversation.instructions

        return await self.create_response(
            input=text,
            instructions=instructions,
        )

    @staticmethod
    def _normalize_message(openai_input: Any) -> List["Message"]:
        from stream_agents.core.agents.conversation import Message

        # standardize on input
        if isinstance(openai_input, str):
            openai_input = [dict(content=openai_input, role="user", type="message")]
        elif not isinstance(openai_input, List):
            openai_input = [openai_input]

        messages: List[Message] = []
        for i in openai_input:
            message = Message(original=i, content=i["content"])
            messages.append(message)

        return messages

    def _tool_schema_to_openai_tool(self, schema) -> Dict[str, Any]:
        """Convert a tool schema to OpenAI tool format."""
        return {
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema.get("description", ""),
                "parameters": schema["parameters_schema"]
            }
        }

    def _prepare_messages(self, input_text: str) -> List[Dict[str, str]]:
        """Convert input text to messages format for chat completions."""
        if isinstance(input_text, str):
            return [{"role": "user", "content": input_text}]
        elif isinstance(input_text, list):
            return input_text
        else:
            return [{"role": "user", "content": str(input_text)}]

    def _normalize_openai_response(self, response) -> NormalizedResponse:
        """Convert OpenAI response to normalized format."""
        output = []
        
        # Handle text content
        message = response.choices[0].message
        if message.content:
            output.append({
                "type": "text",
                "text": message.content
            })
        
        # Handle tool calls if present
        if message.tool_calls:
            for tool_call in message.tool_calls:
                # Parse arguments from JSON string
                import json
                arguments_dict = json.loads(tool_call.function.arguments)
                
                tool_call_item: NormalizedToolCallItem = {
                    "type": "tool_call",
                    "name": tool_call.function.name,
                    "arguments_json": arguments_dict
                }
                output.append(tool_call_item)  # type: ignore
        
        return {
            "id": response.id,
            "model": response.model,
            "status": "completed",  # type: ignore
            "output": output,  # type: ignore
            "output_text": message.content or "",
            "raw": response
        }
    
    async def _generate_conversational_response(self, tool_results: list, original_response: NormalizedResponse) -> str:
        """Generate a conversational response based on tool results using OpenAI."""
        try:
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
            tool_calls = []
            for i, result in enumerate(tool_results):
                tool_calls.append({
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": result["name"],
                        "arguments": "{}"  # We don't need the original arguments
                    }
                })
            
            messages.append({
                "role": "assistant",
                "content": None,  # type: ignore
                "tool_calls": tool_calls  # type: ignore
            })
            
            # Add the tool results
            for i, result in enumerate(tool_results):
                messages.append({
                    "role": "tool",
                    "tool_call_id": f"call_{i}",
                    "content": json.dumps(result["result_json"])
                })
            
            # Add a system message to guide the response
            messages.insert(0, {
                "role": "system",
                "content": "You are a helpful assistant. The user asked you to perform some functions, and you have the results. Please provide a natural, conversational response based on these results. Be helpful and informative."
            })
            
            # Get a follow-up response from OpenAI
            follow_up_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore
                max_tokens=500,
                temperature=0.7
            )
            
            return follow_up_response.choices[0].message.content or ""
            
        except Exception as e:
            # If there's an error, return empty string to use fallback formatting
            print(f"Error generating conversational response: {e}")
            return ""
