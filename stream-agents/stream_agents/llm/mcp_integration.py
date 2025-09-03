"""
MCP (Model Context Protocol) Integration for Stream Agents

This module provides integration with MCP servers, allowing LLMs to use
external tools and data sources through the standardized MCP protocol.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

try:
    from mcp import ClientSession, StdioServerParameters, stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    # Create dummy classes for type hints when MCP is not available
    class ClientSession:
        pass
    class StdioServerParameters:
        pass

# Import FunctionDefinition locally to avoid circular imports
try:
    from .function_registry import FunctionDefinition
except ImportError:
    # Define a minimal FunctionDefinition if import fails
    from dataclasses import dataclass
    from typing import Callable, Any, Optional
    
    @dataclass
    class FunctionDefinition:
        name: str
        description: str
        parameters: dict
        handler: Callable
        is_async: bool
        return_type: Optional[type] = None


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection."""
    name: str
    command: str
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    auth_token: Optional[str] = None


class MCPIntegration:
    """Integration with MCP servers for tool discovery and execution."""
    
    def __init__(self, config: MCPServerConfig):
        """Initialize MCP integration with server configuration."""
        if not MCP_AVAILABLE:
            raise ImportError(
                "MCP package not available. Install with: pip install mcp"
            )
        
        self.config = config
        self.session: Optional[ClientSession] = None
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{config.name}")
        self._connected = False
    
    async def connect(self) -> None:
        """Connect to the MCP server and discover available tools."""
        if self._connected:
            return
        
        try:
            # Create server parameters
            server_params = StdioServerParameters(
                command=self.config.command,
                args=self.config.args or [],
                env=self.config.env or {}
            )
            
            # Connect to server
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    self.session = session
                    
                    # Initialize the session
                    await session.initialize()
                    
                    # Discover available tools
                    await self._discover_tools()
                    
                    self._connected = True
                    self.logger.info(f"Connected to MCP server '{self.config.name}' with {len(self.tools)} tools")
                    
        except Exception as e:
            self.logger.error(f"Failed to connect to MCP server '{self.config.name}': {e}")
            raise
    
    async def _discover_tools(self) -> None:
        """Discover available tools from the MCP server."""
        if not self.session:
            raise RuntimeError("Not connected to MCP server")
        
        try:
            # List available tools
            tools_response = await self.session.list_tools()
            
            # Store tool definitions
            for tool in tools_response.tools:
                self.tools[tool.name] = {
                    "name": tool.name,
                    "description": tool.description or "",
                    "inputSchema": tool.inputSchema or {}
                }
            
            self.logger.debug(f"Discovered {len(self.tools)} tools: {list(self.tools.keys())}")
            
        except Exception as e:
            self.logger.error(f"Failed to discover tools from MCP server: {e}")
            raise
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool call via the MCP server."""
        if not self._connected:
            await self.connect()
        
        if not self.session:
            raise RuntimeError("Not connected to MCP server")
        
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found in MCP server '{self.config.name}'")
        
        try:
            # Call the tool
            result = await self.session.call_tool(name, arguments)
            
            # Extract content from result
            if result.content:
                # Handle different content types
                content = result.content[0]
                if hasattr(content, 'text'):
                    return content.text
                elif hasattr(content, 'data'):
                    return content.data
                else:
                    return str(content)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to call tool '{name}': {e}")
            raise
    
    def get_tool_definitions(self) -> List[FunctionDefinition]:
        """Get FunctionDefinition objects for all discovered tools."""
        definitions = []
        
        for tool_name, tool_info in self.tools.items():
            # Convert MCP schema to our format
            parameters = self._convert_mcp_schema_to_json_schema(tool_info["inputSchema"])
            
            definition = FunctionDefinition(
                name=f"mcp_{self.config.name}_{tool_name}",
                description=tool_info["description"],
                parameters=parameters,
                handler=self._create_tool_handler(tool_name),
                is_async=True,
                return_type=str
            )
            definitions.append(definition)
        
        return definitions
    
    def _convert_mcp_schema_to_json_schema(self, mcp_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert MCP input schema to JSON schema format."""
        if not mcp_schema:
            return {"type": "object", "properties": {}}
        
        # MCP schemas are already in JSON Schema format, but we need to ensure
        # they're compatible with our function registry
        json_schema = mcp_schema.copy()
        
        # Ensure required fields are present
        if "type" not in json_schema:
            json_schema["type"] = "object"
        
        if "properties" not in json_schema:
            json_schema["properties"] = {}
        
        return json_schema
    
    def _create_tool_handler(self, tool_name: str):
        """Create a handler function for an MCP tool."""
        async def handler(**kwargs) -> str:
            result = await self.call_tool(tool_name, kwargs)
            return str(result) if result is not None else ""
        
        return handler
    
    def has_tool(self, name: str) -> bool:
        """Check if the MCP server has a specific tool."""
        return name in self.tools
    
    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self.session:
            # The session will be closed automatically when exiting the context manager
            self.session = None
            self._connected = False
            self.logger.info(f"Disconnected from MCP server '{self.config.name}'")


class MCPManager:
    """Manager for multiple MCP server integrations."""
    
    def __init__(self):
        self.integrations: Dict[str, MCPIntegration] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def add_server(self, config: MCPServerConfig) -> None:
        """Add an MCP server configuration."""
        if config.name in self.integrations:
            raise ValueError(f"MCP server '{config.name}' already exists")
        
        integration = MCPIntegration(config)
        self.integrations[config.name] = integration
        self.logger.info(f"Added MCP server configuration: {config.name}")
    
    async def connect_all(self) -> None:
        """Connect to all configured MCP servers."""
        for name, integration in self.integrations.items():
            try:
                await integration.connect()
            except Exception as e:
                self.logger.error(f"Failed to connect to MCP server '{name}': {e}")
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on any connected MCP server."""
        # Parse tool name to find the right server
        if tool_name.startswith("mcp_"):
            parts = tool_name.split("_", 2)
            if len(parts) >= 3:
                server_name = parts[1]
                actual_tool_name = parts[2]
                
                if server_name in self.integrations:
                    return await self.integrations[server_name].call_tool(actual_tool_name, arguments)
        
        raise ValueError(f"Tool '{tool_name}' not found in any MCP server")
    
    def get_all_tool_definitions(self) -> List[FunctionDefinition]:
        """Get all tool definitions from all MCP servers."""
        definitions = []
        for integration in self.integrations.values():
            definitions.extend(integration.get_tool_definitions())
        return definitions
    
    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        for integration in self.integrations.values():
            await integration.disconnect()
