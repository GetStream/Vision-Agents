"""
WebSocket client implementation for Stream Video Coordinator.

This module contains the StreamAPIWS class that provides an asynchronous WebSocket client
for communicating with Stream's Video Coordinator.
"""

import asyncio
import json
import logging
import time
from typing import Optional

import websockets

from getstream.utils import StreamAsyncIOEventEmitter
from getstream.video.rtc.coordinator.errors import (
    StreamWSAuthError,
    StreamWSConnectionError,
    StreamWSMaxRetriesExceeded,
)
from getstream.video.rtc.coordinator.backoff import exp_backoff

logger = logging.getLogger(__name__)

DEFAULT_WS_URI = "wss://chat.stream-io-api.com/api/v2/connect"


class StreamAPIWS(StreamAsyncIOEventEmitter):
    """
    Asynchronous WebSocket client for Stream Video Coordinator.

    This client handles authentication, event dispatching, and connection management
    for communicating with Stream's Video Coordinator WebSocket API.
    """

    def __init__(
        self,
        api_key: str,
        token: str,
        user_details: Optional[dict] = None,
        *,
        uri: str = DEFAULT_WS_URI,
        healthcheck_interval: float = 15.0,
        healthcheck_timeout: float = 30.0,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        backoff_factor: float = 2.0,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the WebSocket client.

        Args:
            api_key: Stream API key
            token: JWT token for authentication
            user_details: Optional user details dict for the connection
            uri: WebSocket URI to connect to
            healthcheck_interval: Interval in seconds between heartbeat messages
            healthcheck_timeout: Timeout in seconds to wait for server messages
            max_retries: Maximum number of reconnection attempts
            backoff_base: Base delay for exponential backoff
            backoff_factor: Factor for exponential backoff
            logger: Optional logger instance
        """
        super().__init__()

        self.api_key = api_key
        self.token = token
        self.user_details = user_details
        self.uri = f"{uri}?api_key={api_key}&stream-auth-type=jwt"
        self.healthcheck_interval = healthcheck_interval
        self.healthcheck_timeout = healthcheck_timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_factor = backoff_factor
        self._logger = logger

        # Connection state
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._client_id: Optional[str] = None

        # Background tasks
        self._reader_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Heartbeat tracking
        self._last_received = 0.0
        self._reconnect_in_progress = False
        self._initial_connection = True  # Track if this is the first connection

    def _build_auth_payload(self) -> dict:
        """
        Build the authentication payload to send after connection.

        Returns:
            Authentication payload as a dictionary
        """
        payload = {
            "token": self.token,
            "products": ["video"],
        }

        # Include user_details if available (both for initial connection and reconnections)
        if self.user_details:
            payload["user_details"] = self.user_details

        return payload

    async def _open_socket(self) -> dict:
        """
        Open WebSocket connection and perform authentication.

        Returns:
            The first message received from server

        Raises:
            StreamWSAuthError: If authentication fails
            StreamWSConnectionError: If connection fails
        """
        self._logger.debug("Opening WebSocket connection", extra={"uri": self.uri})

        # Open WebSocket connection with ping disabled (we handle heartbeats ourselves)
        self._websocket = await websockets.connect(
            self.uri,
            # ping_interval=None,
            # ping_timeout=None,
        )
        self._logger.debug("WebSocket connection established")

        # Send authentication payload immediately
        auth_payload = self._build_auth_payload()
        self._logger.debug("Sending auth payload")
        await self._websocket.send(json.dumps(auth_payload))

        # Wait for the first server message
        self._logger.debug("Waiting for first server message")
        raw_message = await self._websocket.recv()
        message = json.loads(raw_message)

        # Check if authentication failed
        if message.get("type") in ("error", "connection.error"):
            self._logger.error("Authentication failed", extra={"error": message})
            # Emit the error event before raising exception so wildcard handlers can catch it
            event_type = message.get("type", "unknown")
            self.emit(event_type, message)
            await self._websocket.close()
            self._websocket = None
            raise StreamWSAuthError(f"Authentication failed: {message}")

        # Authentication successful - update last received time and emit event
        self._last_received = time.time()

        # Extract and store the client_id (connection_id) from the response
        self._client_id = message.get("connection_id")
        if self._client_id:
            self._logger.debug("Stored client_id", extra={"client_id": self._client_id})
        else:
            self._logger.warning("No connection_id found in server response")

        self._logger.info("Successfully connected and authenticated")

        # Emit the event
        event_type = message.get("type", "unknown")
        self.emit(event_type, message)

        return message

    async def _reader_task_func(self) -> None:
        """
        Background task that reads messages from the WebSocket and emits events.
        """
        self._logger.debug("Starting reader task")

        try:
            while self._connected and self._websocket:
                try:
                    raw_message = await self._websocket.recv()
                    self._logger.debug("Received message")

                    # Update last received timestamp
                    self._last_received = time.time()

                    # Parse and emit message
                    try:
                        message = json.loads(raw_message)
                        event_type = message.get("type", "unknown")
                        self._logger.debug(
                            "Received message", extra={"type": event_type}
                        )
                        self.emit(event_type, message)
                    except json.JSONDecodeError as e:
                        self._logger.warning(
                            "Failed to parse message as JSON", exc_info=e
                        )

                except websockets.exceptions.ConnectionClosed as e:
                    # Extract close information safely
                    close_code = None
                    close_reason = None
                    try:
                        if hasattr(e, "rcvd") and e.rcvd:
                            close_code = getattr(e.rcvd, "code", None)
                            close_reason = getattr(e.rcvd, "reason", None)
                    except AttributeError:
                        pass

                    self._logger.error(
                        "WebSocket connection closed by server",
                        extra={
                            "close_code": close_code,
                            "close_reason": close_reason,
                        },
                    )
                    if self._connected and not self._reconnect_in_progress:
                        await self._trigger_reconnect()
                    break
                except websockets.exceptions.WebSocketException as e:
                    self._logger.error(
                        "WebSocket protocol error in reader task", exc_info=e
                    )
                    if self._connected and not self._reconnect_in_progress:
                        await self._trigger_reconnect()
                    break
                except Exception as e:
                    self._logger.error("Unexpected error in reader task", exc_info=e)
                    if self._connected and not self._reconnect_in_progress:
                        await self._trigger_reconnect()
                    break

        except asyncio.CancelledError:
            self._logger.debug("Reader task cancelled")
            raise
        finally:
            self._logger.debug("Reader task ended")

    async def _heartbeat_task_func(self) -> None:
        """
        Background task that sends heartbeat messages and monitors connection health.
        """
        self._logger.debug("Starting heartbeat task")

        try:
            while self._connected:
                await asyncio.sleep(self.healthcheck_interval)

                if not self._connected:
                    break

                current_time = time.time()

                # Check if we haven't received anything for too long
                if current_time - self._last_received > self.healthcheck_timeout:
                    self._logger.warning(
                        "No messages received within timeout period",
                        extra={
                            "last_received": self._last_received,
                            "timeout": self.healthcheck_timeout,
                            "elapsed": current_time - self._last_received,
                        },
                    )
                    if not self._reconnect_in_progress:
                        await self._trigger_reconnect()
                    break

                # Send heartbeat
                if self._websocket and self._connected:
                    try:
                        heartbeat = {"type": "health.check"}
                        if self._client_id:
                            heartbeat["client_id"] = self._client_id
                        await self._websocket.send(json.dumps(heartbeat))
                        self._logger.debug("Sent heartbeat")
                    except websockets.exceptions.ConnectionClosed as e:
                        # Extract close information safely
                        close_code = None
                        close_reason = None
                        try:
                            if hasattr(e, "rcvd") and e.rcvd:
                                close_code = getattr(e.rcvd, "code", None)
                                close_reason = getattr(e.rcvd, "reason", None)
                        except AttributeError:
                            pass

                        self._logger.error(
                            "WebSocket connection closed while sending heartbeat",
                            extra={
                                "close_code": close_code,
                                "close_reason": close_reason,
                            },
                        )
                        if not self._reconnect_in_progress:
                            await self._trigger_reconnect()
                        break
                    except websockets.exceptions.WebSocketException as e:
                        self._logger.error(
                            "WebSocket protocol error while sending heartbeat",
                            exc_info=e,
                        )
                        if not self._reconnect_in_progress:
                            await self._trigger_reconnect()
                        break
                    except Exception as e:
                        self._logger.error(
                            "Unexpected error while sending heartbeat", exc_info=e
                        )
                        if not self._reconnect_in_progress:
                            await self._trigger_reconnect()
                        break

        except asyncio.CancelledError:
            self._logger.debug("Heartbeat task cancelled")
            raise
        finally:
            self._logger.debug("Heartbeat task ended")

    async def _trigger_reconnect(self) -> None:
        """
        Trigger reconnection process.
        """
        if self._reconnect_in_progress:
            return

        self._logger.warning("Triggering reconnection")
        self._reconnect_in_progress = True

        try:
            await self._reconnect()
        except Exception as e:
            self._logger.error("Reconnection failed completely", exc_info=e)
            self._connected = False
        finally:
            self._reconnect_in_progress = False

    async def _reconnect(self) -> None:
        """
        Attempt to reconnect using exponential backoff strategy.

        Raises:
            StreamWSMaxRetriesExceeded: If all retry attempts fail
        """
        self._logger.info("Starting reconnection process")

        # Ensure this is not treated as initial connection
        self._initial_connection = False

        # Cancel existing tasks
        await self._cancel_background_tasks()

        # Close existing websocket
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception:
                pass
            self._websocket = None

        # Attempt reconnection with exponential backoff
        attempt = 0
        async for delay in exp_backoff(
            max_retries=self.max_retries,
            base=self.backoff_base,
            factor=self.backoff_factor,
        ):
            attempt += 1
            self._logger.info(
                "Reconnection attempt",
                extra={
                    "attempt": attempt,
                    "max_retries": self.max_retries,
                    "delay": delay,
                },
            )

            await asyncio.sleep(delay)

            if not self._connected:
                self._logger.debug(
                    "Connection was closed during backoff, aborting reconnection"
                )
                return

            try:
                await self._open_socket()

                # Restart background tasks
                await self._start_background_tasks()

                self._logger.info("Reconnection successful")
                return

            except (OSError, websockets.exceptions.WebSocketException) as e:
                self._logger.warning(
                    "Reconnection attempt failed",
                    extra={"attempt": attempt, "error": str(e)},
                )
                continue
            except StreamWSAuthError as e:
                # Authentication errors should not trigger retry
                self._logger.error(
                    "Authentication failed during reconnection", exc_info=e
                )
                self._connected = False
                raise
            except Exception as e:
                self._logger.error("Unexpected error during reconnection", exc_info=e)
                continue

        # All retry attempts exhausted
        self._logger.error("All reconnection attempts failed")
        self._connected = False
        raise StreamWSMaxRetriesExceeded(
            "Maximum number of reconnection attempts exceeded"
        )

    async def _start_background_tasks(self) -> None:
        """Start the background tasks for reading and heartbeat."""
        self._logger.debug("Starting background tasks")

        self._reader_task = asyncio.create_task(self._reader_task_func())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_task_func())

    async def _cancel_background_tasks(self) -> None:
        """Cancel the background tasks."""
        self._logger.debug("Cancelling background tasks")

        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

    async def connect(self) -> dict:
        """
        Open the socket, authenticate, and wait for the first server frame.

        Returns:
            The first event payload (usually {"type": "connection.ok"})

        Raises:
            StreamWSAuthError: If the first frame has type == "error" or authentication times out
            StreamWSConnectionError: If there are connection issues
        """
        self._logger.info("Connecting to coordinator", extra={"uri": self.uri})

        if self._connected:
            self._logger.warning("Already connected, disconnecting first")
            await self.disconnect()

        # Mark this as initial connection
        self._initial_connection = True

        try:
            first_message = await self._open_socket()

            # Mark as connected and start background tasks
            self._connected = True
            await self._start_background_tasks()

            # After successful initial connection, subsequent connections are reconnections
            self._initial_connection = False

            # Return the authentication response
            return first_message

        except StreamWSAuthError:
            # Re-raise authentication errors without modification
            raise
        except websockets.exceptions.WebSocketException as e:
            self._logger.error("WebSocket connection failed", exc_info=e)
            if self._websocket:
                await self._websocket.close()
                self._websocket = None
            raise StreamWSConnectionError(f"WebSocket connection failed: {e}") from e
        except json.JSONDecodeError as e:
            self._logger.error("Failed to parse server message", exc_info=e)
            if self._websocket:
                await self._websocket.close()
                self._websocket = None
            raise StreamWSConnectionError(f"Invalid JSON from server: {e}") from e
        except Exception as e:
            self._logger.error("Unexpected error during connection", exc_info=e)
            if self._websocket:
                await self._websocket.close()
                self._websocket = None
            raise StreamWSConnectionError(f"Unexpected connection error: {e}") from e

    async def disconnect(self) -> None:
        """
        Cancel internal tasks and close the WebSocket cleanly.
        """
        self._logger.info("Disconnecting from coordinator")

        self._connected = False
        self._reconnect_in_progress = False
        self._client_id = None
        self._initial_connection = True  # Reset for next connection

        # Cancel background tasks
        await self._cancel_background_tasks()

        # Close WebSocket
        if self._websocket:
            try:
                await self._websocket.close(code=1000, reason="client disconnect")
                self._logger.debug("WebSocket closed cleanly")
            except Exception as e:
                self._logger.warning("Error closing WebSocket", exc_info=e)
            finally:
                self._websocket = None

        self._logger.info("Disconnected from coordinator")

    @property
    def connected(self) -> bool:
        """
        Check if the WebSocket is currently connected.

        Returns:
            True if connected, False otherwise
        """
        return self._connected and self._websocket is not None
