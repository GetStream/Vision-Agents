from __future__ import annotations

import abc
from typing import Optional, TYPE_CHECKING

from pyee.asyncio import AsyncIOEventEmitter

if TYPE_CHECKING:
    from stream_agents.core.agents import Agent
    from stream_agents.core.agents.conversation import Conversation


from typing import List, TypeVar, Any, Callable, Generic, Dict, Optional as TypingOptional


from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant
from stream_agents.core.processors import BaseProcessor
from stream_agents.core.utils.utils import parse_instructions
from .function_registry import FunctionRegistry
from .llm_types import ToolSchema, NormalizedResponse, NormalizedToolResultItem

T = TypeVar("T")


class LLMResponse(Generic[T]):
    def __init__(self, original: T, text: str):
        self.original = original
        self.text = text


BeforeCb = Callable[[List[Any]], None]
AfterCb = Callable[[LLMResponse], None]


class LLM(AsyncIOEventEmitter, abc.ABC):
    # if we want to use realtime/ sts behaviour
    sts: bool = False

    before_response_listener: BeforeCb
    after_response_listener: AfterCb
    agent: Optional["Agent"]
    _conversation: Optional["Conversation"]
    function_registry: FunctionRegistry

    def __init__(self):
        super().__init__()
        self.agent = None
        self.function_registry = FunctionRegistry()

    async def simple_response(
        self,
        text: str,
        processors: TypingOptional[List[BaseProcessor]] = None,
        participant: TypingOptional[Participant] = None,
    ) -> LLMResponse[Any]:
        raise NotImplementedError

    def _attach_agent(self, agent: Agent):
        """
        Attach agent to the llm
        """
        self.agent = agent
        self._conversation = agent.conversation
        self.instructions = agent.instructions
        
        # Parse instructions to extract @ mentioned markdown files
        self.parsed_instructions = parse_instructions(agent.instructions)

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
    
    async def process_tool_calls(self, response: NormalizedResponse) -> NormalizedResponse:
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
                # Type assertion: we know this is a tool call item
                function_name = item["name"]  # type: ignore
                arguments = item["arguments_json"]  # type: ignore
                
                # arguments_json is now a Dict[str, Any], no need to parse
                
                try:
                    # Call the function
                    result = self.call_function(function_name, arguments)
                    
                    # Add tool result to output
                    tool_result_item: NormalizedToolResultItem = {
                        "type": "tool_result",
                        "name": function_name,
                        "result_json": result if isinstance(result, dict) else {"result": result},
                        "is_error": False
                    }
                    updated_output.append(tool_result_item)
                    
                except Exception as e:
                    # Add error result to output
                    error_result_item: NormalizedToolResultItem = {
                        "type": "tool_result",
                        "name": function_name,
                        "result_json": {"error": str(e)},
                        "is_error": True
                    }
                    updated_output.append(error_result_item)
            else:
                # Keep non-tool-call items as-is
                updated_output.append(item)  # type: ignore
        
        # Update the response with processed output
        response["output"] = updated_output  # type: ignore
        
        # Check if we have tool results that need a conversational response
        tool_results = [item for item in updated_output if item.get("type") == "tool_result"]
        if tool_results:
            # Generate a conversational response based on the tool results
            conversational_response = await self._generate_conversational_response(tool_results, response)
            if conversational_response:
                response["output_text"] = conversational_response
                response["output"] = [{"type": "text", "text": conversational_response}]
            else:
                # Fallback: simple formatting
                response_text_parts = []
                for item in updated_output:
                    if item.get("type") == "text":
                        response_text_parts.append(item["text"])  # type: ignore
                    elif item.get("type") == "tool_result":
                        result = item["result_json"]  # type: ignore
                        function_name = item["name"]  # type: ignore
                        if item.get("is_error", False):
                            response_text_parts.append(f"Error in {function_name}: {result.get('error', 'Unknown error')}")
                        else:
                            response_text_parts.append(f"{function_name} result: {result}")
                response["output_text"] = "\n".join(response_text_parts)
        else:
            # No tool results, just use the text content
            response_text_parts = []
            for item in updated_output:
                if item.get("type") == "text":
                    response_text_parts.append(item["text"])  # type: ignore
            if response_text_parts:
                response["output_text"] = "\n".join(response_text_parts)
        
        return response
    
    async def _generate_conversational_response(self, tool_results: list, original_response: NormalizedResponse) -> str | None:
        """Generate a conversational response based on tool results.
        
        This method should be implemented by each LLM provider to generate
        natural language responses based on function call results.
        
        Args:
            tool_results: List of tool result items
            original_response: The original response containing the tool calls
            
        Returns:
            A conversational response string, or None if not implemented
        """
        # Default implementation returns None to use fallback formatting
        # Each LLM provider should override this method
        return None
