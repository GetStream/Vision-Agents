"""Remote MCP server connection using HTTP Streamable transport."""

from datetime import timedelta
from typing import Optional, Dict, Callable
from urllib.parse import urlparse

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from .mcp_base import MCPBaseServer


class MCPServerRemote(MCPBaseServer):
    """Remote MCP server connection using HTTP Streamable transport."""

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        session_timeout: float = 300.0,
    ):
        """Initialize the remote MCP server connection.

        Args:
            url: URL of the MCP server (e.g., "http://localhost:8001/mcp")
            headers: Optional HTTP headers to include in requests
            timeout: Connection timeout in seconds
            session_timeout: How long an established MCP session can sit idle with no tool calls, no traffic (in seconds)
        """
        super().__init__(session_timeout)
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout
        self._client_context: Optional[object] = None  # AsyncGeneratorContextManager
        self._session_context: Optional[object] = None  # ClientSession context manager
        self._get_session_id_cb: Optional[Callable[[], Optional[str]]] = None

        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL: {url}")

    async def connect(self) -> None:
        """Connect to the remote MCP server."""
        if self._is_connected:
            self.logger.warning("Already connected to MCP server")
            return

        try:
            self.logger.info(f"Connecting to remote MCP server at {self.url}")

            # Create the HTTP client context
            self._client_context = streamablehttp_client(  # type: ignore[assignment]
                self.url, headers=self.headers, timeout=timedelta(seconds=self.timeout)
            )

            # Enter the context to get the read/write streams and session ID callback
            (
                read,
                write,
                self._get_session_id_cb,
            ) = await self._client_context.__aenter__()  # type: ignore[attr-defined]

            # Create the client session context manager
            self._session_context = ClientSession(read, write)  # type: ignore[assignment]

            # Enter the session context and get the actual session
            self._session = await self._session_context.__aenter__()  # type: ignore[attr-defined]

            # Initialize the connection
            await self._session.initialize()

            self._is_connected = True
            await self._update_activity()
            await self._start_timeout_monitor()

            # Log session ID if available
            if self._get_session_id_cb is not None:
                try:
                    session_id = self._get_session_id_cb()
                    self.logger.info(
                        f"Successfully connected to remote MCP server at {self.url} (session: {session_id})"
                    )
                except Exception as e:
                    self.logger.info(
                        f"Successfully connected to remote MCP server at {self.url} (session ID unavailable: {e})"
                    )
            else:
                self.logger.info(
                    f"Successfully connected to remote MCP server at {self.url}"
                )

        except Exception as e:
            self.logger.error(f"Failed to connect to remote MCP server: {e}")
            # Clean up any partial connection state
            await self._cleanup_connection()
            raise

    async def disconnect(self) -> None:
        """Disconnect from the remote MCP server."""
        if not self._is_connected:
            return

        try:
            self.logger.info("Disconnecting from remote MCP server")

            # Stop timeout monitoring
            await self._stop_timeout_monitor()

            # Clean up the connection
            await self._cleanup_connection()

            self._is_connected = False
            self.logger.info("Disconnected from remote MCP server")

        except Exception as e:
            self.logger.error(f"Error disconnecting from remote MCP server: {e}")
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
        """String representation of the remote MCP server."""
        return f"MCPServerRemote(url='{self.url}', connected={self._is_connected})"
