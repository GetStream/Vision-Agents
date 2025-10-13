"""Local MCP server connection using stdio transport."""

from typing import Optional, Dict, Callable

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

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
        self._server_params: Optional[StdioServerParameters] = None
        self._client_context: Optional[object] = None  # AsyncGeneratorContextManager
        self._session_context: Optional[object] = None  # ClientSession context manager
        self._get_session_id_cb: Optional[Callable[[], Optional[str]]] = None

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

        try:
            self.logger.info(f"Connecting to local MCP server: {self.command}")

            # Create server parameters
            self._server_params = StdioServerParameters(
                command=self._executable, args=self._args, env=self.env
            )

            # Create the stdio client context
            self._client_context = stdio_client(self._server_params)  # type: ignore[assignment]

            # Enter the context to get the read/write streams
            # Note: stdio_client only returns (read, write), no session ID callback
            read, write = await self._client_context.__aenter__()  # type: ignore[attr-defined]

            # Create the client session context manager
            self._session_context = ClientSession(read, write)  # type: ignore[assignment]

            # Enter the session context and get the actual session
            self._session = await self._session_context.__aenter__()  # type: ignore[attr-defined]

            # Initialize the connection
            await self._session.initialize()

            self._is_connected = True
            await self._update_activity()
            await self._start_timeout_monitor()

            self.logger.info(
                f"Successfully connected to local MCP server: {self.command}"
            )

        except Exception as e:
            self.logger.error(f"Failed to connect to local MCP server: {e}")
            # Clean up any partial connection state
            await self._cleanup_connection()
            raise

    async def disconnect(self) -> None:
        """Disconnect from the local MCP server."""
        if not self._is_connected:
            return

        try:
            self.logger.info("Disconnecting from local MCP server")

            # Stop timeout monitoring
            await self._stop_timeout_monitor()

            # Clean up the connection
            await self._cleanup_connection()

            self._is_connected = False
            self.logger.info("Disconnected from local MCP server")

        except Exception as e:
            self.logger.error(f"Error disconnecting from local MCP server: {e}")
            self._is_connected = False

    async def _cleanup_connection(self) -> None:
        """Clean up the MCP connection resources."""
        # Close the session context
        if self._session_context:
            try:
                await self._session_context.__aexit__(None, None, None)  # type: ignore[attr-defined]
            except Exception as e:
                self.logger.warning(f"Error closing MCP session context: {e}")
            self._session_context = None

        # Close the client context
        if self._client_context:
            try:
                await self._client_context.__aexit__(None, None, None)  # type: ignore[attr-defined]
            except Exception as e:
                self.logger.warning(f"Error closing MCP client context: {e}")
            self._client_context = None

        self._session = None
        self._get_session_id_cb = None

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
