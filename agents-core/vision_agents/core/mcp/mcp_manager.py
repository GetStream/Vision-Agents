"""MCP Manager for handling MCP server connections and tool management."""

import logging
from typing import List, Any, Dict
from ..mcp import MCPBaseServer
from ..mcp.tool_converter import MCPToolConverter


class MCPManager:
    """Manages MCP server connections and tool registration for agents."""

    def __init__(
        self,
        mcp_servers: List[MCPBaseServer],
        llm,
        logger: logging.Logger | logging.LoggerAdapter,
    ):
        """Initialize the MCP manager.

        Args:
            mcp_servers: List of MCP servers to manage
            llm: LLM instance for tool registration
            logger: Logger instance for this manager
        """
        self.mcp_servers = mcp_servers
        self.llm = llm
        self.logger = logger

    async def connect_all(self):
        """Connect to all configured MCP servers and register their tools."""
        if not self.mcp_servers:
            return

        self.logger.info(f"🔌 Connecting to {len(self.mcp_servers)} MCP server(s)")

        for i, server in enumerate(self.mcp_servers):
            try:
                self.logger.info(
                    f"  Connecting to MCP server {i + 1}/{len(self.mcp_servers)}: {server.__class__.__name__}"
                )
                await server.connect()
                self.logger.info(
                    f"  ✅ Connected to MCP server {i + 1}/{len(self.mcp_servers)}"
                )

                # Register MCP tools with the LLM's function registry
                await self._register_mcp_tools(i, server)

            except Exception as e:
                self.logger.error(
                    f"  ❌ Failed to connect to MCP server {i + 1}/{len(self.mcp_servers)}: {e}"
                )
                # Continue with other servers even if one fails

    async def disconnect_all(self):
        """Disconnect from all configured MCP servers."""
        if not self.mcp_servers:
            return

        self.logger.info(f"🔌 Disconnecting from {len(self.mcp_servers)} MCP server(s)")

        for i, server in enumerate(self.mcp_servers):
            try:
                self.logger.info(
                    f"  Disconnecting from MCP server {i + 1}/{len(self.mcp_servers)}: {server.__class__.__name__}"
                )
                await server.disconnect()
                self.logger.info(
                    f"  ✅ Disconnected from MCP server {i + 1}/{len(self.mcp_servers)}"
                )
            except Exception as e:
                self.logger.error(
                    f"  ❌ Error disconnecting from MCP server {i + 1}/{len(self.mcp_servers)}: {e}"
                )
                # Continue with other servers even if one fails

    async def get_all_tools(self) -> List[Any]:
        """Get all available tools from all connected MCP servers."""
        tools = []

        for server in self.mcp_servers:
            if server.is_connected:
                try:
                    server_tools = await server.list_tools()
                    tools.extend(server_tools)
                except Exception as e:
                    self.logger.error(
                        f"Error getting tools from MCP server {server.__class__.__name__}: {e}"
                    )

        return tools

    async def call_tool(
        self, server_index: int, tool_name: str, arguments: Dict[str, Any]
    ) -> Any:
        """Call a tool on a specific MCP server.

        Args:
            server_index: Index of the MCP server in the mcp_servers list
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            The result of the tool call
        """
        if server_index >= len(self.mcp_servers):
            raise ValueError(f"Invalid server index: {server_index}")
        server = self.mcp_servers[server_index]
        if not server.is_connected:
            raise RuntimeError(f"MCP server {server_index} is not connected")
        return await server.call_tool(tool_name, arguments)

    async def _register_mcp_tools(self, server_index: int, server: MCPBaseServer):
        """Register tools from an MCP server with the LLM's function registry.

        Args:
            server_index: Index of the MCP server in the mcp_servers list
            server: The connected MCP server
        """
        try:
            # Get tools from the MCP server
            mcp_tools = await server.list_tools()
            self.logger.info(
                f"  📋 Found {len(mcp_tools)} tools from MCP server {server_index + 1}"
            )

            # Register each tool with the function registry
            for tool in mcp_tools:
                try:
                    # Create a wrapper function for the MCP tool
                    tool_wrapper = MCPToolConverter.create_mcp_tool_wrapper(
                        server_index, tool.name, self
                    )

                    # Convert the MCP tool schema to our format
                    tool_schema = MCPToolConverter.mcp_tool_to_tool_schema(tool)

                    # Register with prefix to avoid collisions between servers
                    # and with locally registered functions
                    prefixed_name = f"mcp_{server_index}_{tool.name}"
                    self.llm.function_registry.register(
                        name=prefixed_name,
                        description=tool.description or f"MCP tool: {tool.name}",
                        parameters_schema=tool_schema.get("parameters_schema", {}),
                    )(tool_wrapper)

                    self.logger.info(f"    ✅ Registered tool: {prefixed_name}")

                except Exception as e:
                    self.logger.error(
                        f"    ❌ Failed to register tool {tool.name}: {e}"
                    )
                    # Continue with other tools even if one fails

        except Exception as e:
            self.logger.error(
                f"  ❌ Failed to get tools from MCP server {server_index + 1}: {e}"
            )
