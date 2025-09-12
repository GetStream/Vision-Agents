"""Base class for MCP server connections."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from mcp import ClientSession, types


class MCPBaseServer(ABC):
    """Base class for MCP server connections."""
    
    def __init__(self, session_timeout: float = 300.0):
        """Initialize the base MCP server.
        
        Args:
            session_timeout: How long an established MCP session can sit idle with no tool calls, no traffic (in seconds)
        """
        self.session_timeout = session_timeout
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self._session: Optional[ClientSession] = None
        self._is_connected = False
        self._last_activity = None
        self._timeout_task: Optional[asyncio.Task] = None
        
    @abstractmethod
    async def connect(self) -> None:
        """Connect to the MCP server."""
        pass
        
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        pass
        
    @property
    def is_connected(self) -> bool:
        """Check if the server is connected."""
        return self._is_connected
        
    async def _update_activity(self) -> None:
        """Update the last activity timestamp."""
        self._last_activity = asyncio.get_event_loop().time()
        
    async def _start_timeout_monitor(self) -> None:
        """Start monitoring for session timeout."""
        if self._timeout_task:
            self._timeout_task.cancel()
            
        self._timeout_task = asyncio.create_task(self._timeout_monitor())
        
    async def _timeout_monitor(self) -> None:
        """Monitor for session timeout."""
        while self._is_connected:
            await asyncio.sleep(10)  # Check every 10 seconds
            if self._last_activity and self._is_connected:
                idle_time = asyncio.get_event_loop().time() - self._last_activity
                if idle_time > self.session_timeout:
                    self.logger.warning(f"Session timeout after {idle_time:.1f}s of inactivity")
                    await self.disconnect()
                    break
                    
    async def _stop_timeout_monitor(self) -> None:
        """Stop the timeout monitor."""
        if self._timeout_task:
            self._timeout_task.cancel()
            self._timeout_task = None
            
    async def list_tools(self) -> List[types.Tool]:
        """List available tools from the MCP server."""
        if not self._session or not self._is_connected:
            raise RuntimeError("Not connected to MCP server")
            
        await self._update_activity()
        response = await self._session.list_tools()
        return response.tools
        
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> types.CallToolResult:
        """Call a tool on the MCP server.
        
        Args:
            name: Name of the tool to call
            arguments: Arguments to pass to the tool
            
        Returns:
            The result of the tool call
        """
        if not self._session or not self._is_connected:
            raise RuntimeError("Not connected to MCP server")
            
        await self._update_activity()
        return await self._session.call_tool(name, arguments)
        
    async def list_resources(self) -> List[types.Resource]:
        """List available resources from the MCP server."""
        if not self._session or not self._is_connected:
            raise RuntimeError("Not connected to MCP server")
            
        await self._update_activity()
        response = await self._session.list_resources()
        return response.resources
        
    async def read_resource(self, uri: str) -> types.ReadResourceResult:
        """Read a resource from the MCP server.
        
        Args:
            uri: URI of the resource to read
            
        Returns:
            The resource content
        """
        if not self._session or not self._is_connected:
            raise RuntimeError("Not connected to MCP server")
            
        await self._update_activity()
        return await self._session.read_resource(uri)
        
    async def list_prompts(self) -> List[types.Prompt]:
        """List available prompts from the MCP server."""
        if not self._session or not self._is_connected:
            raise RuntimeError("Not connected to MCP server")
            
        await self._update_activity()
        response = await self._session.list_prompts()
        return response.prompts
        
    async def get_prompt(self, name: str, arguments: Dict[str, Any]) -> types.GetPromptResult:
        """Get a prompt from the MCP server.
        
        Args:
            name: Name of the prompt to get
            arguments: Arguments for the prompt
            
        Returns:
            The prompt content
        """
        if not self._session or not self._is_connected:
            raise RuntimeError("Not connected to MCP server")
            
        await self._update_activity()
        return await self._session.get_prompt(name, arguments)
