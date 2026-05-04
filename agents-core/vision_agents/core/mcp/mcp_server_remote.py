"""Remote MCP server connection using HTTP Streamable transport."""

import asyncio
import contextlib
from datetime import timedelta
from typing import Optional, Dict, Callable
from urllib.parse import urlparse

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from ..utils.utils import cancel_and_wait

from .mcp_base import MCPBaseServer


@contextlib.asynccontextmanager
async def _open_mcp_session(url: str, headers: Dict[str, str], timeout_seconds: float):
    """Open the streamable-HTTP transport plus a ``ClientSession`` on top of it
    and yield the ready session together with its session-id callback."""
    async with streamablehttp_client(
        url,
        headers=headers,
        timeout=timedelta(seconds=timeout_seconds),
    ) as (read, write, get_session_id_cb):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session, get_session_id_cb


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

        self._supervisor_task: Optional[asyncio.Task] = None
        self._setup: Optional[asyncio.Future[None]] = None
        self._stop_event: Optional[asyncio.Event] = None
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

        self.logger.info(f"Connecting to remote MCP server at {self.url}")
        self._setup = asyncio.get_running_loop().create_future()
        self._stop_event = asyncio.Event()
        self._supervisor_task = asyncio.create_task(
            self._supervise_session(), name=f"mcp-supervisor:{self.url}"
        )
        try:
            await self._setup
        except (Exception, asyncio.CancelledError):
            # Setup failed, or connect() itself was cancelled. Either way,
            # tear the supervisor down and let its cleanup finish before
            # propagating. Anything else (SystemExit, KeyboardInterrupt) is
            # left to propagate immediately without our cleanup detour.
            await self._teardown_supervisor()
            raise

    async def disconnect(self) -> None:
        """Disconnect from the remote MCP server."""
        if self._supervisor_task is None:
            return
        self.logger.info("Disconnecting from remote MCP server")
        await self._teardown_supervisor()
        self.logger.info("Disconnected from remote MCP server")

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
        """Hold the MCP session open until ``_stop_event`` is set."""
        if self._setup is None or self._stop_event is None:
            raise RuntimeError(
                "_supervise_session must be started by connect(); "
                "_setup or _stop_event is not initialized"
            )
        try:
            async with _open_mcp_session(self.url, self.headers, self.timeout) as (
                session,
                get_session_id_cb,
            ):
                self._session = session
                self._get_session_id_cb = get_session_id_cb
                self._is_connected = True
                await self._update_activity()
                await self._start_timeout_monitor()
                self._log_connection_success()

                self._setup.set_result(None)
                await self._stop_event.wait()
        except Exception as e:
            if not self._setup.done():
                self._setup.set_exception(e)
                self.logger.exception("Failed to connect to remote MCP server")
            else:
                self.logger.warning(
                    "MCP session supervisor exited with error", exc_info=True
                )
        finally:
            # Make sure connect() never hangs even if we were cancelled mid-setup.
            if not self._setup.done():
                self._setup.cancel()
            try:
                await self._stop_timeout_monitor()
            except Exception:
                self.logger.warning("Error stopping timeout monitor", exc_info=True)
            self._session = None
            self._get_session_id_cb = None
            self._is_connected = False

    def _log_connection_success(self) -> None:
        """Log a connection-success line, including the MCP session id if known."""
        msg = f"Successfully connected to remote MCP server at {self.url}"
        if self._get_session_id_cb is not None:
            try:
                msg += f" (session: {self._get_session_id_cb()})"
            except Exception as e:
                msg += f" (session ID unavailable: {e})"
                self.logger.debug("Session ID lookup failed", exc_info=True)
        self.logger.info(msg)

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
