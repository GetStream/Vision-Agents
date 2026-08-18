import json
import logging
from typing import Any, AsyncIterator, Optional, Union

import aiohttp

logger = logging.getLogger(__name__)


class Socket:
    """One WebSocket to the router.

    The rest of the client is generated from the OpenAPI spec, but OpenAPI stops at the
    upgrade, so the sockets are written by hand. This is the whole of it: JSON one way,
    JSON and audio the other.
    """

    def __init__(self, url: str, headers: dict[str, str]):
        self._url = url
        self._headers = headers
        self._session: Optional[aiohttp.ClientSession] = None
        self._connection: Optional[aiohttp.ClientWebSocketResponse] = None

    @property
    def open(self) -> bool:
        """Whether the socket can still carry a message."""
        return self._connection is not None and not self._connection.closed

    async def connect(self) -> None:
        """Open the socket, raising if the router refuses it."""
        self._session = aiohttp.ClientSession()
        self._connection = await self._session.ws_connect(
            self._url, headers=self._headers, autoping=True
        )

    async def send(self, frame: dict[str, Any]) -> None:
        """Send one JSON frame."""
        if self._connection is None:
            raise RuntimeError("the socket is not open")
        await self._connection.send_str(json.dumps(frame))

    async def send_audio(self, payload: bytes) -> None:
        """Send one binary frame."""
        if self._connection is None:
            raise RuntimeError("the socket is not open")
        await self._connection.send_bytes(payload)

    async def frames(self) -> AsyncIterator[Union[dict[str, Any], bytes]]:
        """Yield what the router sends until the socket closes.

        Text frames arrive decoded and binary frames as they are. A text frame that is not
        JSON is dropped: it can only be a bug on the far side, and stopping the stream over
        it would lose everything said after.
        """
        if self._connection is None:
            raise RuntimeError("the socket is not open")

        async for message in self._connection:
            if message.type == aiohttp.WSMsgType.TEXT:
                try:
                    yield json.loads(message.data)
                except json.JSONDecodeError:
                    logger.warning("dropping a frame that is not JSON")
            elif message.type == aiohttp.WSMsgType.BINARY:
                yield message.data
            elif message.type == aiohttp.WSMsgType.ERROR:
                logger.warning("the socket failed: %s", self._connection.exception())
                break

    async def close(self) -> None:
        """Close the socket and the session behind it. Safe to call twice."""
        if self._connection is not None and not self._connection.closed:
            await self._connection.close()
        self._connection = None
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None
