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
        self.logger = logging.getLogger(__name__)
        self._session: Optional[ClientSession] = None
        self._is_connected = False
        self._last_activity: Optional[float] = None
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
                    self.logger.warning(
                        f"Session timeout after {idle_time:.1f}s of inactivity"
                    )
                    await self.disconnect()
                    break

    async def _stop_timeout_monitor(self) -> None:
        """Stop the timeout monitor."""
        if self._timeout_task:
            self._timeout_task.cancel()
            self._timeout_task = None

    async def _ensure_connected(self) -> None:
        """Ensure the server is connected, reconnecting if necessary."""
        if not self._is_connected:
            self.logger.info("Reconnecting to MCP server...")
            await self.connect()

    async def _call_with_retry(
        self, operation_name: str, operation_func, *args, **kwargs
    ):
        """Call an MCP operation with auto-reconnect on failure."""
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                await self._ensure_connected()
                await self._update_activity()
                return await operation_func(*args, **kwargs)
            except Exception as e:
                if attempt < max_retries:
                    self.logger.warning(
                        f"{operation_name} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. Reconnecting..."
                    )
                    await self.disconnect()
                    await asyncio.sleep(1)  # Brief delay before retry
                else:
                    self.logger.error(
                        f"{operation_name} failed after {max_retries + 1} attempts: {e}"
                    )
                    raise

    async def _list_tools_impl(self) -> List[types.Tool]:
        """Internal implementation of list_tools without retry logic."""
        if not self._session or not self._is_connected:
            raise RuntimeError("Not connected to MCP server")

        await self._update_activity()
        response = await self._session.list_tools()
        return response.tools

    async def _call_tool_impl(
        self, name: str, arguments: Dict[str, Any]
    ) -> types.CallToolResult:
        """Internal implementation of call_tool without retry logic."""
        if not self._session or not self._is_connected:
            raise RuntimeError("Not connected to MCP server")

        await self._update_activity()
        return await self._session.call_tool(name, arguments)

    async def _list_resources_impl(self) -> List[types.Resource]:
        """Internal implementation of list_resources without retry logic."""
        if not self._session or not self._is_connected:
            raise RuntimeError("Not connected to MCP server")

        await self._update_activity()
        response = await self._session.list_resources()
        return response.resources

    async def _read_resource_impl(self, uri: str) -> types.ReadResourceResult:
        """Internal implementation of read_resource without retry logic."""
        if not self._session or not self._is_connected:
            raise RuntimeError("Not connected to MCP server")

        await self._update_activity()
        from mcp.types import AnyUrl

        return await self._session.read_resource(AnyUrl(uri))

    async def _list_prompts_impl(self) -> List[types.Prompt]:
        """Internal implementation of list_prompts without retry logic."""
        if not self._session or not self._is_connected:
            raise RuntimeError("Not connected to MCP server")

        await self._update_activity()
        response = await self._session.list_prompts()
        return response.prompts

    async def _get_prompt_impl(
        self, name: str, arguments: Dict[str, Any]
    ) -> types.GetPromptResult:
        """Internal implementation of get_prompt without retry logic."""
        if not self._session or not self._is_connected:
            raise RuntimeError("Not connected to MCP server")

        await self._update_activity()
        return await self._session.get_prompt(name, arguments)

    async def list_tools(self) -> List[types.Tool]:
        """List available tools from the MCP server with auto-reconnect."""
        return await self._call_with_retry("list_tools", self._list_tools_impl)

    async def call_tool(
        self, name: str, arguments: Dict[str, Any]
    ) -> types.CallToolResult:
        """Call a tool on the MCP server with auto-reconnect."""
        return await self._call_with_retry(
            "call_tool", self._call_tool_impl, name, arguments
        )

    async def list_resources(self) -> List[types.Resource]:
        """List available resources from the MCP server with auto-reconnect."""
        return await self._call_with_retry("list_resources", self._list_resources_impl)

    async def read_resource(self, uri: str) -> types.ReadResourceResult:
        """Read a resource from the MCP server with auto-reconnect."""
        return await self._call_with_retry(
            "read_resource", self._read_resource_impl, uri
        )

    async def list_prompts(self) -> List[types.Prompt]:
        """List available prompts from the MCP server with auto-reconnect."""
        return await self._call_with_retry("list_prompts", self._list_prompts_impl)

    async def get_prompt(
        self, name: str, arguments: Dict[str, Any]
    ) -> types.GetPromptResult:
        """Get a prompt from the MCP server with auto-reconnect."""
        return await self._call_with_retry(
            "get_prompt", self._get_prompt_impl, name, arguments
        )
