"""Utility to convert MCP tools to function registry format."""

from typing import Any, Dict, Callable
from mcp import types

from ..llm.llm_types import ToolSchema


class MCPToolConverter:
    """Converts MCP tools to function registry format."""

    @staticmethod
    def mcp_tool_to_tool_schema(tool: types.Tool) -> ToolSchema:
        """Convert an MCP tool to a ToolSchema.

        Args:
            tool: MCP tool object

        Returns:
            ToolSchema compatible with function registry
        """
        # Convert MCP tool input schema to JSON schema format
        parameters_schema = MCPToolConverter._convert_input_schema(tool.inputSchema)

        return ToolSchema(
            name=tool.name,
            description=tool.description or "",
            parameters_schema=parameters_schema,
        )

    @staticmethod
    def _convert_input_schema(input_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert MCP input schema to JSON schema format.

        Args:
            input_schema: MCP tool input schema

        Returns:
            JSON schema compatible with function registry
        """
        # MCP tools already use JSON schema format, so we can mostly pass through
        # but we need to ensure it has the right structure
        schema = input_schema.copy()

        # Ensure required fields are present
        if "type" not in schema:
            schema["type"] = "object"

        if "properties" not in schema:
            schema["properties"] = {}

        # Ensure additionalProperties is set
        if "additionalProperties" not in schema:
            schema["additionalProperties"] = False

        return schema

    @staticmethod
    def create_mcp_tool_wrapper(
        server_index: int, tool_name: str, agent_ref
    ) -> "Callable":
        """Create a wrapper function for calling MCP tools.

        Args:
            server_index: Index of the MCP server in the agent's mcp_servers list
            tool_name: Name of the tool to call
            agent_ref: Reference to the agent instance

        Returns:
            Callable function that can be registered with the function registry
        """

        async def mcp_tool_wrapper(**kwargs) -> Any:
            """Wrapper function for MCP tool calls."""
            try:
                result = await agent_ref.call_tool(server_index, tool_name, kwargs)
                # Extract the actual result from MCP response
                if hasattr(result, "content") and result.content:
                    # MCP tools return CallToolResult with content
                    if isinstance(result.content, list) and len(result.content) > 0:
                        # Get the first content item
                        content_item = result.content[0]
                        if hasattr(content_item, "text"):
                            return content_item.text
                        elif hasattr(content_item, "data"):
                            return content_item.data
                        else:
                            return str(content_item)
                    else:
                        return str(result.content)
                else:
                    return str(result)
            except Exception as e:
                # Return error information in a structured way
                return {
                    "error": str(e),
                    "tool": tool_name,
                    "server_index": server_index,
                }

        return mcp_tool_wrapper
