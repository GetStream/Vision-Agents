import asyncio
import logging
from dataclasses import dataclass
from os import getenv
from typing import Callable, Coroutine
from uuid import uuid4

import av
from getstream.video.rtc.track_util import AudioFormat, PcmData
from livekit import api, rtc
from PIL import Image
from vision_agents.core.utils.utils import cancel_and_wait

logger = logging.getLogger(__name__)

_AUDIO_STREAM_TOPIC = "lk.audio_stream"
_AVATAR_IDENTITY = "avatar"
_PLUGIN_IDENTITY = "plugin"
_SAMPLE_RATE = 16000
_NUM_CHANNELS = 1


@dataclass(frozen=True)
class Credentials:
    """Credentials for LemonSlice to join the LiveKit room."""

    url: str
    token: str


class LemonSliceRTCManager:
    """Manages a LiveKit room connection for LemonSlice avatar streaming.

    Creates a LiveKit room, sends TTS audio to LemonSlice via data streams,
    and receives synchronized avatar video and audio tracks.
    """

    def __init__(
        self,
        on_video: Callable[[av.VideoFrame], Coroutine[None, None, None]],
        on_audio: Callable[[PcmData], Coroutine[None, None, None]],
        on_disconnect: Callable[[], Coroutine[None, None, None]],
        livekit_url: str | None = None,
        livekit_api_key: str | None = None,
        livekit_api_secret: str | None = None,
    ):
        self._livekit_url = livekit_url or getenv("LIVEKIT_URL") or ""
        if not self._livekit_url:
            raise ValueError(
                "LiveKit URL required. Set LIVEKIT_URL environment variable "
                "or pass livekit_url parameter."
            )

        self._livekit_api_key = livekit_api_key or getenv("LIVEKIT_API_KEY") or ""
        self._livekit_api_secret = (
            livekit_api_secret or getenv("LIVEKIT_API_SECRET") or ""
        )
        if not self._livekit_api_key or not self._livekit_api_secret:
            raise ValueError(
                "LiveKit API key and secret required. Set LIVEKIT_API_KEY and "
                "LIVEKIT_API_SECRET environment variables or pass them as parameters."
            )

        self._on_video = on_video
        self._on_audio = on_audio
        self._on_disconnect = on_disconnect

        self._room: rtc.Room | None = None
        self._stream_writer: rtc.ByteStreamWriter | None = None
        self._connected = False
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> Credentials:
        """Connect to a LiveKit room.

        Returns:
            Credentials for the LemonSlice participant to join the same room.
        """
        room_name = f"lemonslice-{uuid4()}"
        agent_token = self._generate_token(room_name, _PLUGIN_IDENTITY, kind="agent")
        lemonslice_token = self._generate_token(
            room_name, _AVATAR_IDENTITY, kind="agent"
        )

        room = rtc.Room()

        @room.on("connected")
        def on_connected():
            logger.info("Room connected")

        @room.on("participant_connected")
        def on_participant_connected(participant: rtc.RemoteParticipant):
            if participant.identity == _AVATAR_IDENTITY:
                logger.info("LemonSlice avatar entered the room")

        @room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            if participant.identity == _AVATAR_IDENTITY:
                if track.kind == rtc.TrackKind.KIND_VIDEO:
                    logger.info("Received video track from LemonSlice")
                    video_stream = rtc.VideoStream(track)
                    self._create_task(self._consume_video(video_stream))
                elif track.kind == rtc.TrackKind.KIND_AUDIO:
                    logger.info("Received audio track from LemonSlice")
                    audio_stream = rtc.AudioStream(
                        track, sample_rate=48000, num_channels=2
                    )
                    self._create_task(self._consume_audio(audio_stream))

        @room.on("participant_disconnected")
        def on_participant_disconnected(participant: rtc.RemoteParticipant) -> None:
            logger.info(
                f"Participant disconnected: {participant.identity}; "
                f"reason: {participant.disconnect_reason}"
            )
            self._connected = False
            self._create_task(self._on_disconnect())
            if self._room is not None:
                self._create_task(self._room.disconnect())

        @room.on("disconnected")
        def on_disconnected(reason: str) -> None:
            logger.info(f"Room disconnected; reason: {reason}")
            self._connected = False
            self._create_task(self._on_disconnect())

        logger.info(f"Connecting to LiveKit room {room_name}")
        await room.connect(self._livekit_url, agent_token)
        logger.info(f"Connected to LiveKit room {room_name}")

        room.local_participant.register_rpc_method(
            "lk.playback_finished", self._rpc_on_playback_finished
        )

        self._room = room
        self._connected = True

        return Credentials(url=self._livekit_url, token=lemonslice_token)

    async def send_audio(self, pcm: PcmData) -> None:
        """Send a PCM audio chunk to LemonSlice via a LiveKit byte stream.

        Args:
            pcm: Audio data to send. Resampled to 16 kHz mono automatically.
        """
        if self._room is None or not self._room.isconnected():
            return

        if pcm.sample_rate != _SAMPLE_RATE or pcm.channels != _NUM_CHANNELS:
            pcm = pcm.resample(
                target_sample_rate=_SAMPLE_RATE,
                target_channels=_NUM_CHANNELS,
            )

        if self._stream_writer is None:
            self._stream_writer = await self._room.local_participant.stream_bytes(
                name=f"AUDIO_{uuid4()}",
                topic=_AUDIO_STREAM_TOPIC,
                destination_identities=[_AVATAR_IDENTITY],
                attributes={
                    "sample_rate": str(pcm.sample_rate),
                    "num_channels": str(pcm.channels),
                },
            )
            logger.debug("Opened audio byte stream to LemonSlice")

        await self._stream_writer.write(pcm.to_bytes())

    async def flush(self) -> None:
        """Close the current byte stream, signalling end of a TTS segment."""
        if self._stream_writer is not None:
            await self._stream_writer.aclose()
            self._stream_writer = None
            logger.debug("Closed audio byte stream (segment end)")

    async def close(self) -> None:
        """Disconnect from the LiveKit room and clean up resources."""
        try:
            if self._stream_writer is not None:
                await self._stream_writer.aclose()

            await cancel_and_wait(*self._tasks)
            self._tasks.clear()

            if self._room is not None:
                await self._room.disconnect()
        finally:
            self._room = None
            self._stream_writer = None
            self._connected = False
            logger.debug("LemonSlice RTC manager closed")

    async def _consume_video(self, video_stream: rtc.VideoStream) -> None:
        async for event in video_stream:
            lk_frame = event.frame.convert(rtc.VideoBufferType.RGBA)
            img = Image.frombuffer(
                "RGBA", (lk_frame.width, lk_frame.height), lk_frame.data
            )
            frame = av.VideoFrame.from_image(img)
            await self._on_video(frame)

    async def _consume_audio(self, audio_stream: rtc.AudioStream) -> None:
        async for event in audio_stream:
            frame = event.frame
            pcm = PcmData.from_bytes(
                frame.data,  # type: ignore[arg-type]
                sample_rate=frame.sample_rate,
                format=AudioFormat.S16,
                channels=frame.num_channels,
            )
            await self._on_audio(pcm)

    def _rpc_on_playback_finished(self, data: rtc.RpcInvocationData) -> str:
        logger.info(
            "playback finished event received",
            extra={"caller_identity": data.caller_identity},
        )
        return "ok"

    def _generate_token(
        self,
        room_name: str,
        identity: str,
        kind: api.AccessToken.ParticipantKind,
    ) -> str:
        token = (
            api.AccessToken(self._livekit_api_key, self._livekit_api_secret)
            .with_kind(kind)
            .with_identity(identity)
            .with_name(identity)
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_publish=True,
                    can_subscribe=True,
                )
            )
        )
        return token.to_jwt()

    def _create_task(self, coro: Coroutine[None, None, None]) -> None:
        task: asyncio.Task[None] = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
