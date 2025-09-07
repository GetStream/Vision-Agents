from __future__ import annotations

from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from stream_agents.agents import Agent

from typing import List, TypeVar, Optional, Any, Callable, Generic, Dict
import logging

from av.dictionary import Dictionary

from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant
from stream_agents.processors import BaseProcessor

T = TypeVar("T")

class LLMResponse(Generic[T]):
    def __init__(self, original: T, text: str):
        self.original = original
        self.text = text

BeforeCb = Callable[[List[Dictionary]], None]
AfterCb  = Callable[[LLMResponse], None]

try:
    from .mcp_integration import MCPManager  # noqa: F401
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


class LLM:
    # if we want to use realtime/ sts behaviour
    sts: bool = False

    before_response_listener: BeforeCb
    after_response_listener: AfterCb
    agent: Optional["Agent"]
    _conversation: Optional["Conversation"]


    def __init__(self):
        self.agent = None
        # Function calling support
        from .function_registry import FunctionRegistry
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

    # Function calling methods
    def function(self, description: str = "", name: Optional[str] = None):
        """
        Decorator to register a function for LLM calling.
        
        Args:
            description: Human-readable description of what the function does
            name: Optional custom name for the function (defaults to function name)
        
        Example:
            @llm.function("Get weather for a location")
            async def get_weather(location: str) -> str:
                return f"Weather in {location}: sunny"
        """
        return self.function_registry.function(description, name)
    
    async def generate_with_functions(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> str:
        """
        Generate response with function calling support.
        
        This method should be implemented by each LLM subclass to handle
        function calling in a provider-specific way.
        """
        raise NotImplementedError("Subclasses must implement generate_with_functions")
    
    def get_available_functions(self) -> List[str]:
        """Get list of available function names."""
        return self.function_registry.list_functions()
    
    def get_function_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered function."""
        return self.function_registry.get_function_info(name)
    
    # MCP methods
    def add_mcp_server(self, config) -> None:
        """
        Add an MCP server to the LLM's function registry.
        
        Args:
            config: MCP server configuration
        
        Example:
            config = MCPServerConfig(
                name="filesystem",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/directory"]
            )
            llm.add_mcp_server(config)
        """
        if not MCP_AVAILABLE:
            raise ImportError(
                "MCP package not available. Install with: pip install mcp"
            )
        
        self.function_registry.add_mcp_server(config)
    
    async def connect_mcp_servers(self) -> None:
        """Connect to all configured MCP servers and register their tools."""
        await self.function_registry.connect_mcp_servers()



