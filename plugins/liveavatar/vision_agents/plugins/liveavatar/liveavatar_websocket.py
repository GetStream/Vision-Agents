import asyncio
import base64
import json
import logging
import uuid

import websockets
from getstream.video.rtc.track_util import PcmData
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)


class LiveAvatarWebSocket:
    """Audio bridge to the LiveAvatar media server (LITE-mode events)."""

    def __init__(
        self,
        ws_url: str,
        sample_rate: int = 24000,
        num_channels: int = 1,
    ) -> None:
        self._ws_url = ws_url
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._ws: ClientConnection | None = None
        self._closed = False
        self._reconnect_lock = asyncio.Lock()

    @property
    def connected(self) -> bool:
        return self._ws is not None

    async def connect(self) -> None:
        if self.connected:
            return
        # ping_interval=None: continuous audio frames are themselves a
        # liveness signal; the LiveAvatar media server stalls pong responses
        # under load and the client tears down the conn (1011) otherwise.
        self._ws = await websockets.connect(self._ws_url, ping_interval=None)
        await self._send_json(
            {
                "type": "start",
                "encoding": "pcm_s16le",
                "sample_rate": self._sample_rate,
                "channels": self._num_channels,
            }
        )
        logger.info("liveavatar_ws connected url=%s", self._ws_url)

    async def close(self) -> None:
        self._closed = True
        if self._ws is not None:
            try:
                await self._ws.close()
            except ConnectionClosed:
                pass
            self._ws = None

    async def send_audio_frame(self, pcm: PcmData) -> None:
        pcm = pcm.resample(
            target_sample_rate=self._sample_rate,
            target_channels=self._num_channels,
        )
        b64 = base64.b64encode(pcm.to_bytes()).decode("ascii")
        await self._send_json({"type": "agent.speak", "audio": b64})

    async def end_turn(self) -> None:
        await self._send_json({"type": "agent.speak_end"})

    async def interrupt(self) -> None:
        await self._send_json(
            {"type": "agent.interrupt", "event_id": str(uuid.uuid4())}
        )

    async def _send_json(self, msg: dict[str, object]) -> None:
        if self._closed:
            raise RuntimeError("liveavatar_ws is closed")
        if not self.connected:
            await self.connect()
        assert self._ws is not None
        try:
            await self._ws.send(json.dumps(msg))
        except ConnectionClosed:
            logger.warning("liveavatar_ws connection closed during send; reconnecting")
            self._ws = None
            await self._reconnect()
            assert self._ws is not None
            await self._ws.send(json.dumps(msg))

    async def _reconnect(self) -> None:
        async with self._reconnect_lock:
            if self.connected or self._closed:
                return
            await self.connect()
