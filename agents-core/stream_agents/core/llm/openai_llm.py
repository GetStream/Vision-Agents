import datetime
from typing import Optional, List, ParamSpec, TypeVar, Callable, TYPE_CHECKING, Dict, Any

from openai import OpenAI
from openai.resources.responses import Responses

from getstream.models import Response
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant
from stream_agents.core.llm.llm import LLM, LLMResponse
from stream_agents.core.llm.llm_types import NormalizedResponse, NormalizedToolCallItem, NormalizedToolResultItem

from stream_agents.core.processors import BaseProcessor

if TYPE_CHECKING:
    from stream_agents.core.agents.conversation import Message


P = ParamSpec("P")
R = TypeVar("R")

def use_create(fn: Callable[P, R]) -> Callable[P, R]:
    return fn

# TODO: somehow this isn't right, docs aren't great: https://peps.python.org/pep-0612/
bound = use_create(Responses.create)

class OpenAILLM(LLM):
    '''
    The goal is to expose the regular/native openAI SDK methods,
    and only standardize the minimal feature set that's needed for the agent integration.

    The agent requires that we standardize:
    - sharing instructions
    - keeping conversation history
    - response normalization

    Notes on the OpenAI integration
    - the native method is called create_response (maps 1-1 to responses.create)
    - history is maintained using conversation.create()

    TODO:
    - proper typing for args, kwargs
    '''
    def __init__(self, model: str, api_key: Optional[str] = None, client: Optional[OpenAI] = None):
        super().__init__()
        self.model = model
        self.openai_conversation = None
        self.conversation = None


        if client is not None:
            self.client = client
        elif api_key is not None and api_key != "":
            self.openai_conversation = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI()

    async def create_response(self, *args: P.args, **kwargs: P.kwargs) -> LLMResponse:
        if "model" not in kwargs:
            kwargs["model"] = self.model

        if not self.openai_conversation:
            self.openai_conversation = self.client.conversations.create()
        kwargs["conversation"] = self.openai_conversation.id

        # Add tools if functions are available
        available_functions = self.get_available_functions()
        if available_functions:
            kwargs["tools"] = [self._tool_schema_to_openai_tool(schema) for schema in available_functions]

        if hasattr(self, "before_response_listener"):
            self.before_response_listener(self._normalize_message(kwargs["input"]))
        response = self.client.responses.create(
            *args, **kwargs
        )

        # Process tool calls if present
        normalized_response = self._normalize_openai_response(response)
        if normalized_response.get("output"):
            normalized_response = self.process_tool_calls(normalized_response)

        llm_response = LLMResponse[Response](response, response.output_text)
        if hasattr(self, "after_response_listener"):
            await self.after_response_listener(llm_response)
        return llm_response

    async def simple_response(self, text: str, processors: Optional[List[BaseProcessor]] = None, participant: Participant = None):
        instructions = None
        if self.conversation is not None:
            instructions = self.conversation.instructions

        return await self.create_response(
            input=text,
            instructions=instructions,
        )

    @staticmethod
    def _normalize_message(openai_input) -> List["Message"]:
        from stream_agents.core.agents.conversation import Message
        
        # standardize on input
        if isinstance(openai_input, str):
            openai_input = [
                dict(content=openai_input, role="user", type="message")
            ]
        elif not isinstance(openai_input, List):
            openai_input = [openai_input]

        messages: List[Message] = []
        for i in openai_input:
            message = Message(original=i, content = i["content"])
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

    def _normalize_openai_response(self, response: Response) -> NormalizedResponse:
        """Convert OpenAI response to normalized format."""
        output = []
        
        # Handle text content
        if hasattr(response, 'output_text') and response.output_text:
            output.append({
                "type": "text",
                "text": response.output_text
            })
        
        # Handle tool calls if present
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_call_item: NormalizedToolCallItem = {
                    "type": "tool_call",
                    "name": tool_call.function.name,
                    "arguments_json": tool_call.function.arguments
                }
                output.append(tool_call_item)
        
        return {
            "id": getattr(response, 'id', ''),
            "model": getattr(response, 'model', self.model),
            "status": "completed",
            "output": output,
            "output_text": getattr(response, 'output_text', ''),
            "raw": response
        }