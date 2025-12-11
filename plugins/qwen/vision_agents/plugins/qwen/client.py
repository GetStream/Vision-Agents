import asyncio
import base64
import contextlib
import json
import logging
import time
from typing import Any, AsyncIterator, Optional

import websockets
from getstream.video.rtc import PcmData

logger = logging.getLogger(__name__)


class Qwen3RealtimeClient:
    """
    A wrapper around WebSocket connection for Qwen3Realtime API.
    It automatically reconnects in case of connection failures.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        config: dict[str, Any],
        reconnect_backoff: float = 1.0,
    ) -> None:
        self._base_url = f"{base_url}?model={model}"
        self._api_key = api_key
        self._real_ws: Optional[websockets.ClientConnection] = None
        self._exit_stack = contextlib.AsyncExitStack()
        self._config = config
        self._conn_lock = asyncio.Lock()
        self._closed = False
        self._reconnect_backoff = reconnect_backoff

    async def connect(self) -> None:
        if self._conn_lock.locked():
            return None

        async with self._conn_lock:
            logger.debug(f"Connecting to Qwen3Realtime at {self._base_url}")
            self._real_ws = await self._exit_stack.enter_async_context(
                websockets.connect(
                    uri=self._base_url,
                    additional_headers={"Authorization": f"Bearer {self._api_key}"},
                )
            )
            # Initialize session with config params
            await self.update_session(self._config)
        return None

    async def close(self) -> None:
        self._closed = True
        try:
            await self._exit_stack.aclose()
        except Exception as e:
            logger.warning(f"Error closing session: {e}")

    async def read(self) -> AsyncIterator[dict[str, Any]]:
        while not self._closed:
            try:
                async for msg in self._ws:
                    event = json.loads(msg)
                    yield event
            except websockets.ConnectionClosedError as e:
                if not _should_reconnect(e):
                    raise
                await asyncio.sleep(self._reconnect_backoff)
                await self.connect()

    async def send_event(self, event: dict[str, Any]) -> None:
        event["event_id"] = f"event_{int(time.time() * 1000)}"

        try:
            await self._ws.send(json.dumps(event))
        except websockets.ConnectionClosedError as e:
            if not _should_reconnect(e):
                raise
            logger.warning(
                f"Re-establishing Qwen3Realtime connection due to error: {e}"
            )
            await asyncio.sleep(self._reconnect_backoff)
            await self.connect()

    async def update_session(self, config: dict[str, Any]) -> None:
        """Update the session configuration."""
        await self.send_event(event={"type": "session.update", "session": config})

    async def send_audio(self, pcm: PcmData) -> None:
        """Stream raw audio data to the API."""
        # Only 16-bit, 16 kHz, mono PCM is supported.
        audio_bytes = pcm.resample(
            target_sample_rate=16000, target_channels=1
        ).samples.tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode()
        append_event = {"type": "input_audio_buffer.append", "audio": audio_b64}
        await self.send_event(append_event)

    async def commit_audio(self) -> None:
        """Commit the audio buffer to trigger processing."""
        event = {"type": "input_audio_buffer.commit"}
        await self.send_event(event)

    async def send_frame(self, frame_bytes: bytes) -> None:
        """
        Append image data to the image buffer.

        Note:
            - The image format must be JPG or JPEG. A resolution of 480p or 720p is recommended.
                The maximum supported resolution is 1080p.
            - A single image should not exceed 500 KB in size.
            - Encode the image data to Base64 before sending.
            - We recommend sending images to the server at a rate of no more than 2 frames per second.
            - You must send audio data at least once before sending image data.
        """
        image_b64 = base64.b64encode(frame_bytes).decode()
        event = {"type": "input_image_buffer.append", "image": image_b64}
        await self.send_event(event)

    async def cancel_response(self) -> None:
        """Cancel the current response."""
        event = {"type": "response.cancel"}
        await self.send_event(event)

    @property
    def _ws(self) -> websockets.ClientConnection:
        if self._real_ws is None:
            raise ValueError("The websocket connection is not established yet")
        return self._real_ws


def _should_reconnect(exc: Exception) -> bool:
    """
    Temporary errors should typically trigger a reconnect.
    So if the websocket breaks this should return True and trigger a reconnect
    """
    reconnect_close_codes = [
        1011,  # Server-side exception or session timeout
        1012,  # Service restart
        1013,  # Try again later
        1014,  # Bad gateway
    ]
    if (
        isinstance(exc, websockets.ConnectionClosedError)
        and exc.rcvd
        and exc.rcvd.code in reconnect_close_codes
    ):
        return True
    return False
