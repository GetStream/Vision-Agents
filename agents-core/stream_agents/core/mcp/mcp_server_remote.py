"""Remote MCP server connection using HTTP Streamable transport."""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse

from mcp import ClientSession, types
from mcp.client.streamable_http import streamablehttp_client

from .mcp_base import MCPBaseServer


class MCPServerRemote(MCPBaseServer):
    """Remote MCP server connection using HTTP Streamable transport."""
    
    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        session_timeout: float = 300.0
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
        self._client_context = None
        
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
            self._client_context = streamablehttp_client(
                self.url,
                headers=self.headers,
                timeout=self.timeout
            )
            
            # Enter the context to get the read/write streams
            read, write, _ = await self._client_context.__aenter__()
            
            # Create the client session
            self._session = ClientSession(read, write)
            
            # Initialize the connection
            await self._session.initialize()
            
            self._is_connected = True
            await self._update_activity()
            await self._start_timeout_monitor()
            
            self.logger.info(f"Successfully connected to remote MCP server at {self.url}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to remote MCP server: {e}")
            # Clean up any partial connection state
            self._is_connected = False
            self._session = None
            if self._client_context:
                try:
                    await self._client_context.__aexit__(None, None, None)
                except Exception:
                    pass
                finally:
                    self._client_context = None
            raise
            
    async def disconnect(self) -> None:
        """Disconnect from the remote MCP server."""
        if not self._is_connected:
            return
            
        try:
            self.logger.info("Disconnecting from remote MCP server")
            
            # Stop timeout monitoring
            await self._stop_timeout_monitor()
            
            # Close the session
            if self._session:
                # Note: ClientSession doesn't have a close method, but we can clean up our references
                self._session = None
                
            # Exit the client context
            if self._client_context:
                try:
                    await self._client_context.__aexit__(None, None, None)
                except Exception as e:
                    self.logger.debug(f"Error closing client context: {e}")
                finally:
                    self._client_context = None
                    
            self._is_connected = False
            self.logger.info("Disconnected from remote MCP server")
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from remote MCP server: {e}")
            self._is_connected = False
            
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
