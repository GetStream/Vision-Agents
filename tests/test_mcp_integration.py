import asyncio
import os
import pytest

from vision_agents.core.mcp.mcp_server_local import MCPServerLocal
from vision_agents.core.mcp.mcp_server_remote import MCPServerRemote
from vision_agents.core.agents import Agent
from vision_agents.core.edge.types import User
from vision_agents.plugins.openai.openai_llm import OpenAILLM
from vision_agents.plugins import getstream, elevenlabs, deepgram
from dotenv import load_dotenv

load_dotenv()


def get_mcp_server():
    """Get configured MCP server based on environment variables."""
    local_cmd = os.getenv("MCP_LOCAL_CMD")
    remote_url = os.getenv("MCP_REMOTE_URL")

    if not local_cmd and not remote_url:
        pytest.skip(
            "No MCP server configured. Set MCP_LOCAL_CMD or MCP_REMOTE_URL to run this test."
        )

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

    if local_cmd:
        return MCPServerLocal(command=local_cmd, session_timeout=60.0)
    else:
        return MCPServerRemote(
            url=remote_url, headers=headers, timeout=30.0, session_timeout=60.0
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_live_list_and_call_tool():
    """Live MCP integration test - basic tool listing and calling.

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
    server = get_mcp_server()

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
        # The result typically has .content or .data; validate it's something
        # We avoid strict structure assumptions; ensure it's truthy
        assert result is not None

        # 3) Optionally: list resources to ensure the protocol flows
        try:
            resources = await server.list_resources()
            assert isinstance(resources, list)
        except Exception:
            # Not all servers support resources; ignore
            pass


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_multiple_tool_calls():
    """Test calling multiple tools in sequence to verify session stability."""
    server = get_mcp_server()

    async with server:
        tools = await server.list_tools()
        assert len(tools) > 0, "No tools available for testing"

        # Test calling multiple tools
        for i, tool in enumerate(tools[:3]):  # Test up to 3 tools
            args = {}
            if tool.name == "get_forecast":
                args = {"city": f"City{i}"}
            else:
                # Generic arguments for unknown tools
                for key in ("query", "q", "text", "input", "name", "message"):
                    args[key] = f"test_{i}"
                    break

            result = await server.call_tool(tool.name, args)
            assert result is not None, f"Tool {tool.name} returned None"

            # Verify we can still list tools after each call
            tools_after = await server.list_tools()
            assert len(tools_after) == len(tools), "Tool count changed unexpectedly"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_resources():
    """Test MCP resource listing and reading functionality."""
    server = get_mcp_server()

    async with server:
        # List resources
        try:
            resources = await server.list_resources()
            assert isinstance(resources, list), "Resources should be a list"

            # If resources are available, try to read one
            if resources:
                resource = resources[0]
                result = await server.read_resource(resource.uri)
                assert result is not None, f"Failed to read resource {resource.uri}"

        except Exception as e:
            # Not all servers support resources
            pytest.skip(f"Resource functionality not supported: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_concurrent_calls():
    """Test making concurrent MCP tool calls using GitHub server."""
    github_pat = os.getenv("GITHUB_PAT")
    if not github_pat:
        pytest.skip("GITHUB_PAT not set, skipping concurrent calls test")

    # Use GitHub MCP server for concurrent testing
    server = MCPServerRemote(
        url="https://api.githubcopilot.com/mcp/",
        headers={"Authorization": f"Bearer {github_pat}"},
        timeout=30.0,
        session_timeout=60.0,
    )

    async with server:
        tools = await server.list_tools()
        if len(tools) < 2:
            pytest.skip("Need at least 2 tools for concurrent testing")

        # Make concurrent calls to different tools
        async def call_tool(tool, args):
            return await server.call_tool(tool.name, args)

        tasks = []
        for i, tool in enumerate(tools[:3]):  # Test up to 3 concurrent calls
            # Use appropriate arguments for GitHub tools
            if tool.name == "search_repositories":
                args = {"query": f"python concurrent_test_{i}"}
            elif tool.name == "get_repository":
                args = {"owner": "microsoft", "repo": "vscode"}
            elif tool.name == "list_issues":
                args = {"owner": "microsoft", "repo": "vscode", "state": "open"}
            else:
                # Generic arguments for unknown GitHub tools
                args = {"query": f"concurrent_test_{i}"}

            tasks.append(call_tool(tool, args))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all calls succeeded
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pytest.fail(f"Concurrent call {i} failed: {result}")
            assert result is not None, f"Concurrent call {i} returned None"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_tool_schema_validation():
    """Test that tool schemas are properly structured."""
    server = get_mcp_server()

    async with server:
        tools = await server.list_tools()
        assert len(tools) > 0, "No tools available for schema testing"

        for tool in tools:
            # Verify tool has required fields
            assert hasattr(tool, "name"), f"Tool missing name: {tool}"
            assert hasattr(tool, "description"), f"Tool missing description: {tool}"
            assert hasattr(tool, "inputSchema"), f"Tool missing inputSchema: {tool}"

            # Verify name is not empty
            assert tool.name, f"Tool name is empty: {tool}"

            # Verify inputSchema is a dict
            assert isinstance(tool.inputSchema, dict), (
                f"Tool inputSchema should be dict: {tool.inputSchema}"
            )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_github_integration():
    """Test integration with GitHub MCP server (requires GITHUB_PAT)."""
    github_pat = os.getenv("GITHUB_PAT")
    if not github_pat:
        pytest.skip("GITHUB_PAT not set, skipping GitHub MCP test")

    # Use GitHub MCP server
    server = MCPServerRemote(
        url="https://api.githubcopilot.com/mcp/",
        headers={"Authorization": f"Bearer {github_pat}"},
        timeout=30.0,
        session_timeout=60.0,
    )

    async with server:
        # List GitHub tools
        tools = await server.list_tools()
        assert len(tools) > 0, "No GitHub tools available"

        # Look for common GitHub tools
        tool_names = {tool.name for tool in tools}
        github_tools = [
            "search_repositories",
            "get_repository",
            "list_issues",
            "create_issue",
        ]
        found_github_tools = [name for name in github_tools if name in tool_names]

        assert len(found_github_tools) > 0, (
            f"No expected GitHub tools found. Available: {tool_names}"
        )

        # Test a simple GitHub tool call
        if "search_repositories" in tool_names:
            result = await server.call_tool("search_repositories", {"query": "python"})
            assert result is not None, "GitHub search_repositories returned None"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_weather_integration():
    """Test integration with weather MCP server (local FastMCP example)."""
    local_cmd = os.getenv("MCP_LOCAL_CMD")
    if not local_cmd or "transport.py" not in local_cmd:
        pytest.skip("MCP_LOCAL_CMD not set to weather server, skipping weather test")

    server = MCPServerLocal(command=local_cmd, session_timeout=60.0)

    async with server:
        tools = await server.list_tools()
        assert len(tools) > 0, "No weather tools available"

        # Look for weather-specific tools
        tool_names = {tool.name for tool in tools}
        if "get_forecast" in tool_names:
            # Test weather forecast tool
            result = await server.call_tool("get_forecast", {"city": "London"})
            assert result is not None, "Weather forecast returned None"

            # Verify result contains weather information
            if hasattr(result, "content") and result.content:
                content_str = str(result.content)
                assert len(content_str) > 0, "Weather forecast content is empty"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openai_llm_mcp_weather_integration():
    """Test OpenAI LLM integration with MCP weather server.

    This test verifies the complete flow:
    1. Agent connects to MCP weather server
    2. MCP tools are registered with LLM function registry
    3. LLM makes function calls to MCP tools
    4. Tool results are processed and returned

    Requires:
    - OPENAI_API_KEY environment variable
    - MCP_LOCAL_CMD pointing to weather server
    - STREAM_API_KEY and STREAM_API_SECRET environment variables
    """
    # Skip if credentials not available
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set, skipping OpenAI MCP integration test")
    if not os.getenv("MCP_LOCAL_CMD") or "transport.py" not in os.getenv(
        "MCP_LOCAL_CMD", ""
    ):
        pytest.skip("MCP_LOCAL_CMD not set to weather server, skipping test")
    if not os.getenv("STREAM_API_KEY") or not os.getenv("STREAM_API_SECRET"):
        pytest.skip("STREAM_API_KEY or STREAM_API_SECRET not set, skipping test")

    # Setup components
    llm = OpenAILLM(
        model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY")
    )  # Use cheaper model
    weather_server = MCPServerLocal(
        command=os.getenv("MCP_LOCAL_CMD"), session_timeout=60.0
    )

    # Create real edge and agent user
    edge = getstream.Edge()
    agent_user = User(name="Weather Assistant", id="weather-agent")

    # Create agent with required processing capabilities
    agent = Agent(
        edge=edge,
        llm=llm,
        agent_user=agent_user,
        instructions="You are a helpful weather assistant. Use the weather tool to get current weather information.",
        mcp_servers=[weather_server],
        tts=elevenlabs.TTS(),
        stt=deepgram.STT(),
    )

    try:
        # Connect to MCP server
        await agent._connect_mcp_servers()

        # Verify tools are registered
        available_functions = agent.llm.get_available_functions()
        mcp_functions = [f for f in available_functions if f["name"].startswith("mcp_")]
        assert len(mcp_functions) > 0, "No MCP tools registered"

        # Test function calling
        response = await agent.llm.simple_response(
            text="What's the weather like in London?",
        )

        # Verify response was received (the core integration test)
        assert response is not None, "No response received from LLM"

    finally:
        await agent.close()
