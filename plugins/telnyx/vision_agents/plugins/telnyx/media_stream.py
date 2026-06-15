"""Telnyx Media Streaming WebSocket handler."""

import base64
import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

from getstream.video import rtc
from getstream.video.rtc.audio_track import AudioStreamTrack
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import TrackType
from getstream.video.rtc.track_util import PcmData
from getstream.video.rtc.tracks import SubscriptionConfig, TrackSubscriptionConfig

from .audio import (
    TELNYX_DEFAULT_SAMPLE_RATE,
    pcm_to_telnyx_payload,
    telnyx_payload_to_pcm,
)

logger = logging.getLogger(__name__)


class WebSocketProtocol(Protocol):
    """Protocol for WebSocket connections compatible with FastAPI and Starlette."""

    async def accept(self) -> None: ...
    async def receive_text(self) -> str: ...
    async def send_json(self, data: Any) -> None: ...


@dataclass
class TelnyxMediaFormat:
    encoding: str = "PCMU"
    sample_rate: int = TELNYX_DEFAULT_SAMPLE_RATE
    channels: int = 1

    @classmethod
    def from_start_event(cls, data: dict[str, Any]) -> "TelnyxMediaFormat":
        media_format = data.get("start", {}).get("media_format", {})
        return cls(
            encoding=str(media_format.get("encoding", "PCMU")).upper(),
            sample_rate=int(
                media_format.get("sample_rate", TELNYX_DEFAULT_SAMPLE_RATE)
            ),
            channels=int(media_format.get("channels", 1)),
        )


class TelnyxMediaStream:
    """
    Manages a Telnyx Media Streaming WebSocket connection.
    """

    def __init__(
        self,
        websocket: WebSocketProtocol,
        *,
        track: str = "inbound",
        media_format: TelnyxMediaFormat | None = None,
    ):
        self.websocket = websocket
        self.track = track
        self.stream_id: str | None = None
        self.call_control_id: str | None = None
        self.media_format = media_format or TelnyxMediaFormat()
        self.audio_track = self._create_audio_track()
        self._connected = False
        self._started = False
        self._start_callbacks: list[Callable[[], Awaitable[None]]] = []
        self._cleanup_callbacks: list[Callable[[], Awaitable[None]]] = []
        self._cleanup_done = False

    async def accept(self) -> None:
        await self.websocket.accept()
        self._connected = True
        logger.info("TelnyxMediaStream: WebSocket connection accepted")

    async def run(self) -> None:
        """
        Process incoming Telnyx WebSocket messages until the stream ends.
        """
        has_seen_media = False
        message_count = 0

        try:
            while True:
                message = await self.websocket.receive_text()
                data = json.loads(message)

                match data["event"]:
                    case "connected":
                        logger.info("TelnyxMediaStream: Connected: %s", data)
                    case "start":
                        self.stream_id = data.get("stream_id")
                        self.call_control_id = data.get("start", {}).get(
                            "call_control_id"
                        )
                        await self._handle_start(
                            TelnyxMediaFormat.from_start_event(data)
                        )
                        logger.info(
                            "TelnyxMediaStream: Stream started, stream_id=%s, encoding=%s",
                            self.stream_id,
                            self.media_format.encoding,
                        )
                    case "media":
                        media = data["media"]
                        if not self._track_matches(str(media.get("track", ""))):
                            continue

                        payload = base64.b64decode(media["payload"])
                        pcm = telnyx_payload_to_pcm(
                            payload,
                            self.media_format.encoding,
                            sample_rate=self.media_format.sample_rate,
                        )
                        await self.audio_track.write(pcm)

                        if not has_seen_media:
                            logger.info(
                                "TelnyxMediaStream: Receiving audio: %s bytes/chunk",
                                len(payload),
                            )
                            has_seen_media = True
                    case "stop":
                        logger.info("TelnyxMediaStream: Stream stopped")
                        break
                    case "error":
                        logger.warning("TelnyxMediaStream: Error frame: %s", data)
                        break
                    case "mark" | "dtmf":
                        logger.debug("TelnyxMediaStream: Event: %s", data)

                message_count += 1
        except Exception:
            logger.exception("TelnyxMediaStream: WebSocket processing failed")
        finally:
            await self.close()

        logger.info(
            "TelnyxMediaStream: Connection closed. Received %s messages",
            message_count,
        )

    async def send_audio(self, pcm: PcmData) -> None:
        """
        Send PCM audio back to Telnyx as a bidirectional RTP media frame.
        """
        if not self._connected or not self._started or self.stream_id is None:
            return

        payload = pcm_to_telnyx_payload(
            pcm,
            self.media_format.encoding,
            sample_rate=self.media_format.sample_rate,
        )
        await self.websocket.send_json(
            {
                "event": "media",
                "media": {"payload": base64.b64encode(payload).decode("ascii")},
            }
        )

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def has_started(self) -> bool:
        return self._started

    def add_start_callback(self, callback: Callable[[], Awaitable[None]]) -> None:
        self._start_callbacks.append(callback)

    def add_cleanup_callback(self, callback: Callable[[], Awaitable[None]]) -> None:
        self._cleanup_callbacks.append(callback)

    async def close(self) -> None:
        self._connected = False
        if self._cleanup_done:
            return
        self._cleanup_done = True

        callbacks = self._cleanup_callbacks
        self._cleanup_callbacks = []
        for callback in callbacks:
            try:
                await callback()
            except Exception:
                logger.exception("TelnyxMediaStream: Cleanup callback failed")

    def _track_matches(self, received: str) -> bool:
        if self.track == "both":
            return True
        if received.endswith("_track"):
            received = received[: -len("_track")]
        return received == self.track

    async def _handle_start(self, media_format: TelnyxMediaFormat) -> None:
        if media_format != self.media_format:
            self.media_format = media_format
            self.audio_track = self._create_audio_track()
        else:
            self.media_format = media_format

        self._started = True
        callbacks = self._start_callbacks
        self._start_callbacks = []
        for callback in callbacks:
            await callback()

    def _create_audio_track(self) -> AudioStreamTrack:
        return AudioStreamTrack(
            sample_rate=self.media_format.sample_rate,
            channels=self.media_format.channels,
            format="s16",
        )


async def attach_phone_to_call(
    call, telnyx_stream: TelnyxMediaStream, user_id: str
) -> None:
    """
    Attach a phone user to a Stream call, bridging audio between Telnyx and Stream.
    """
    subscription_config = SubscriptionConfig(
        default=TrackSubscriptionConfig(track_types=[TrackType.TRACK_TYPE_AUDIO])
    )

    connection = await rtc.join(call, user_id, subscription_config=subscription_config)

    @connection.on("audio")
    async def on_audio_received(pcm: PcmData):
        await telnyx_stream.send_audio(pcm)

    async def add_telnyx_track() -> None:
        await connection.add_tracks(audio=telnyx_stream.audio_track, video=None)

    await connection.__aenter__()
    telnyx_stream.add_cleanup_callback(lambda: connection.__aexit__(None, None, None))
    if telnyx_stream.has_started:
        await add_telnyx_track()
    else:
        telnyx_stream.add_start_callback(add_telnyx_track)

    logger.info("Phone user %s attached to call", user_id)
