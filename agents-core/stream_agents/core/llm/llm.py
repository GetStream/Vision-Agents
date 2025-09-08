from __future__ import annotations

from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from stream_agents.core.agents import Agent
    from stream_agents.core.agents.conversation import Conversation


from typing import List, TypeVar, Optional, Any, Callable, Generic, Dict

from av.dictionary import Dictionary

from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant
from stream_agents.core.processors import BaseProcessor
from .function_registry import FunctionRegistry
from .llm_types import ToolSchema, NormalizedResponse, NormalizedToolCallItem, NormalizedToolResultItem

T = TypeVar("T")

class LLMResponse(Generic[T]):
    def __init__(self, original: T, text: str):
        self.original = original
        self.text = text

BeforeCb = Callable[[List[Dictionary]], None]
AfterCb  = Callable[[LLMResponse], None]



class LLM:
    # if we want to use realtime/ sts behaviour
    sts: bool = False

    before_response_listener: BeforeCb
    after_response_listener: AfterCb
    agent: Optional["Agent"]
    _conversation: Optional[Conversation]
    function_registry: FunctionRegistry

    def __init__(self):
        self.agent = None
        self.function_registry = FunctionRegistry()

    def simple_response(self, text, processors: List[BaseProcessor], participant: Participant = None) -> LLMResponse[Any]:
        pass

    def attach_agent(self, agent: Agent):
        self.agent = agent
        self._conversation = agent.conversation
        self.before_response_listener = lambda x: agent.before_response(x)
        self.after_response_listener = lambda x: agent.after_response(x)

    def set_before_response_listener(self, before_response_listener: BeforeCb):
        self.before_response_listener = before_response_listener

    def set_after_response_listener(self, after_response_listener: AfterCb):
        self.after_response_listener = after_response_listener

    def register_function(self, 
                         name: Optional[str] = None,
                         description: Optional[str] = None) -> Callable:
        """
        Decorator to register a function with the LLM's function registry.
        
        Args:
            name: Optional custom name for the function. If not provided, uses the function name.
            description: Optional description for the function. If not provided, uses the docstring.
        
        Returns:
            Decorator function.
        """
        return self.function_registry.register(name, description)
    
    def get_available_functions(self) -> List[ToolSchema]:
        """Get a list of available function schemas."""
        return self.function_registry.get_tool_schemas()
    
    def call_function(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a registered function with the given arguments.
        
        Args:
            name: Name of the function to call.
            arguments: Dictionary of arguments to pass to the function.
        
        Returns:
            Result of the function call.
        """
        return self.function_registry.call_function(name, arguments)
    
    def process_tool_calls(self, response: NormalizedResponse) -> NormalizedResponse:
        """
        Process tool calls in a response by executing the functions and adding results.
        
        Args:
            response: The normalized response containing tool calls.
        
        Returns:
            Updated response with tool results.
        """
        if not response.get("output"):
            return response
        
        updated_output = []
        
        for item in response["output"]:
            if item.get("type") == "tool_call":
                tool_call_item = item
                function_name = tool_call_item["name"]
                arguments = tool_call_item["arguments_json"]
                
                try:
                    # Call the function
                    result = self.call_function(function_name, arguments)
                    
                    # Add tool result to output
                    tool_result: NormalizedToolResultItem = {
                        "type": "tool_result",
                        "name": function_name,
                        "result_json": result if isinstance(result, dict) else {"result": result},
                        "is_error": False
                    }
                    updated_output.append(tool_result)
                    
                except Exception as e:
                    # Add error result to output
                    tool_result: NormalizedToolResultItem = {
                        "type": "tool_result",
                        "name": function_name,
                        "result_json": {"error": str(e)},
                        "is_error": True
                    }
                    updated_output.append(tool_result)
            else:
                # Keep non-tool-call items as-is
                updated_output.append(item)
        
        # Update the response with processed output
        response["output"] = updated_output
        return response







