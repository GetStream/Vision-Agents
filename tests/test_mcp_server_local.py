import asyncio
import sys
from pathlib import Path

import pytest
from vision_agents.core.mcp import MCPServerLocal

ECHO_SERVER = Path(__file__).with_name("echo_mcp_server.py")


@pytest.fixture
def server() -> MCPServerLocal:
    return MCPServerLocal(command=f"{sys.executable} {ECHO_SERVER}")


class TestMCPServerLocal:
    async def test_a_tool_on_the_server_can_be_called(self, server: MCPServerLocal):
        async with server:
            tools = await server.list_tools()
            result = await server.call_tool("echo", {"text": "hello"})

        assert [tool.name for tool in tools] == ["echo"]
        assert result.content[0].text == "hello"
        assert not server.is_connected

    async def test_disconnecting_from_another_task_leaves_the_caller_running(
        self, server: MCPServerLocal
    ):
        # An agent closes on a task of its own, not the one that connected. The stdio
        # transport's cancel scopes must still be exited where they were entered, or the
        # cancellation lands on whoever connected.
        await server.connect()

        await asyncio.create_task(server.disconnect())
        await asyncio.sleep(0.1)

        assert not server.is_connected

    async def test_a_disconnected_server_can_connect_again(
        self, server: MCPServerLocal
    ):
        await server.connect()
        await server.disconnect()

        async with server:
            result = await server.call_tool("echo", {"text": "again"})

        assert result.content[0].text == "again"
