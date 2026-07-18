import asyncio
import base64
import json
import logging
import uuid

import av
import websockets
from getstream.video.rtc.track_util import FrameResampler, PcmData
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
        self._resampler = FrameResampler(
            rate=sample_rate,
            layout="stereo" if num_channels == 2 else "mono",
            format="s16",
            frame_size=0,
        )

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
        try:
            await self._ws.send(
                json.dumps(
                    {
                        "type": "start",
                        "encoding": "pcm_s16le",
                        "sample_rate": self._sample_rate,
                        "channels": self._num_channels,
                    }
                )
            )
        except ConnectionClosed:
            self._ws = None
            raise
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
        for frame in self._resampler.resample(pcm):
            await self._send_frame(frame)

    async def end_turn(self) -> None:
        # Flush the resampler tail so the utterance plays out, then end the turn.
        for frame in self._resampler.flush():
            await self._send_frame(frame)
        await self._send_json({"type": "agent.speak_end"})

    async def interrupt(self) -> None:
        # Discard the resampler tail so it doesn't bleed into the next turn.
        self._resampler.flush()
        await self._send_json(
            {"type": "agent.interrupt", "event_id": str(uuid.uuid4())}
        )

    async def _send_frame(self, frame: av.AudioFrame) -> None:
        b64 = base64.b64encode(frame.to_ndarray().tobytes()).decode("ascii")
        await self._send_json({"type": "agent.speak", "audio": b64})

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
