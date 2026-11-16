"""Local MCP server connection using stdio transport."""

import asyncio
from typing import Optional, Dict

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ..utils.utils import cancel_and_wait

from .mcp_base import MCPBaseServer


class MCPServerLocal(MCPBaseServer):
    """Local MCP server connection using stdio transport."""

    def __init__(
        self,
        command: str,
        env: Optional[Dict[str, str]] = None,
        session_timeout: float = 300.0,
    ):
        """Initialize the local MCP server connection.

        Args:
            command: Command to run the MCP server (e.g., "python", "node", etc.)
            env: Optional environment variables to pass to the server process
            session_timeout: How long an established MCP session can sit idle with no tool calls, no traffic (in seconds)
        """
        super().__init__(session_timeout)
        self.command = command
        self.env = env or {}
        self._supervisor_task: Optional[asyncio.Task] = None
        self._setup: Optional[asyncio.Future[None]] = None
        self._stop_event: Optional[asyncio.Event] = None

        # Parse command into executable and arguments
        self._parse_command()

    def _parse_command(self) -> None:
        """Parse the command string into executable and arguments."""
        parts = self.command.split()
        if not parts:
            raise ValueError("Command cannot be empty")

        self._executable = parts[0]
        self._args = parts[1:] if len(parts) > 1 else []

    async def connect(self) -> None:
        """Connect to the local MCP server."""
        if self._is_connected:
            self.logger.warning("Already connected to MCP server")
            return

        self.logger.info(f"Connecting to local MCP server: {self.command}")
        self._setup = asyncio.get_running_loop().create_future()
        self._stop_event = asyncio.Event()
        self._supervisor_task = asyncio.create_task(
            self._supervise_session(), name=f"mcp-supervisor:{self._executable}"
        )
        try:
            await self._setup
        except (Exception, asyncio.CancelledError):
            await self._teardown_supervisor()
            raise

    async def disconnect(self) -> None:
        """Disconnect from the local MCP server."""
        if self._supervisor_task is None:
            return
        self.logger.info("Disconnecting from local MCP server")
        await self._teardown_supervisor()
        self.logger.info("Disconnected from local MCP server")

    async def _teardown_supervisor(self) -> None:
        """Signal the supervisor to stop and await its exit."""
        try:
            if self._stop_event is not None:
                self._stop_event.set()
            if self._supervisor_task is not None:
                await cancel_and_wait(self._supervisor_task)
        finally:
            self._supervisor_task = None
            self._setup = None
            self._stop_event = None

    async def _supervise_session(self) -> None:
        """Hold the MCP session open until ``_stop_event`` is set.

        The stdio transport is built on anyio cancel scopes, which must be entered and
        exited in the same task, so this task owns the whole session.
        """
        if self._setup is None or self._stop_event is None:
            raise RuntimeError(
                "_supervise_session must be started by connect(); "
                "_setup or _stop_event is not initialized"
            )
        params = StdioServerParameters(
            command=self._executable, args=self._args, env=self.env
        )
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._is_connected = True
                    await self._update_activity()
                    await self._start_timeout_monitor()
                    self.logger.info(
                        f"Successfully connected to local MCP server: {self.command}"
                    )

                    self._setup.set_result(None)
                    await self._stop_event.wait()
        except Exception as e:
            if not self._setup.done():
                self._setup.set_exception(e)
                self.logger.exception("Failed to connect to local MCP server")
            else:
                self.logger.warning(
                    "MCP session supervisor exited with error", exc_info=True
                )
        finally:
            if not self._setup.done():
                self._setup.cancel()
            try:
                await self._stop_timeout_monitor()
            except Exception:
                self.logger.warning("Error stopping timeout monitor", exc_info=True)
            self._session = None
            self._is_connected = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    def __repr__(self) -> str:
        """String representation of the local MCP server."""
        return (
            f"MCPServerLocal(command='{self.command}', connected={self._is_connected})"
        )
