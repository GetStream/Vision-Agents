import os
import asyncio
import pytest

from stream_agents.core.mcp.mcp_server_local import MCPServerLocal
from stream_agents.core.mcp.mcp_server_remote import MCPServerRemote


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_live_list_and_call_tool():
    """Live MCP integration test.

    Configure via environment:
    - MCP_LOCAL_CMD: if set, runs a local stdio MCP server with this command
      Example: "uv run python examples/plugins_examples/mcp/transport.py"
    - MCP_REMOTE_URL: if set, connects to a remote HTTP MCP server (streamable-http)
      Example: "http://localhost:8001/mcp"
    - MCP_REMOTE_HEADERS_*: optional headers, e.g., MCP_REMOTE_HEADERS_Authorization="Bearer <token>"
    
Cursor says to set this:
export MCP_LOCAL_CMD='uv run python examples/plugins_examples/mcp/transport.py'


    At least one of MCP_LOCAL_CMD or MCP_REMOTE_URL must be provided, otherwise the test is skipped.
    """
    local_cmd = os.getenv("MCP_LOCAL_CMD")
    remote_url = os.getenv("MCP_REMOTE_URL")

    if not local_cmd and not remote_url:
        pytest.skip("No MCP server configured. Set MCP_LOCAL_CMD or MCP_REMOTE_URL to run this test.")

    # Build optional headers for remote
    headers = None
    if remote_url:
        headers = {}
        for k, v in os.environ.items():
            if k.startswith("MCP_REMOTE_HEADERS_") and v:
                hdr_name = k[len("MCP_REMOTE_HEADERS_") :].replace("_", "-")
                headers[hdr_name] = v
        if not headers:
            headers = None

    server = None
    if local_cmd:
        server = MCPServerLocal(command=local_cmd, session_timeout=60.0)
    else:
        server = MCPServerRemote(url=remote_url, headers=headers, timeout=30.0, session_timeout=60.0)

    async with server:
        # 1) List tools
        tools = await server.list_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0, "No tools returned by MCP server"

        # Prefer an obvious tool if present
        tool_names = {t.name for t in tools}
        chosen = None
        preferred = [
            "get_forecast",
            "probe",
            "health",
            "status",
        ]
        for name in preferred:
            if name in tool_names:
                chosen = name
                break
        if not chosen:
            # Fallback: just pick the first tool
            chosen = tools[0].name

        # 2) Call the tool with a generic argument shape
        args = {}
        if chosen == "get_forecast":
            args = {"city": os.getenv("TEST_MCP_CITY", "New York")}
        else:
            # Try a simple echo-ish parameter name patterns to maximize success for generic servers
            for key in ("query", "q", "text", "input", "name"):
                args[key] = "ping"
                break

        result = await server.call_tool(chosen, args)
        # The result typically has .content or .data; validate it’s something
        # We avoid strict structure assumptions; ensure it’s truthy
        assert result is not None

        # 3) Optionally: list resources to ensure the protocol flows
        try:
            resources = await server.list_resources()
            assert isinstance(resources, list)
        except Exception:
            # Not all servers support resources; ignore
            pass
