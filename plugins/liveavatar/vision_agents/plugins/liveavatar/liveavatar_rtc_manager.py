import asyncio
import logging
from typing import Callable, Coroutine

import av
import numpy as np
from getstream.video.rtc.track_util import AudioFormat, PcmData
from livekit import rtc
from vision_agents.core.utils.utils import cancel_and_wait

logger = logging.getLogger(__name__)


VideoCallback = Callable[[av.VideoFrame], Coroutine[None, None, None]]
AudioCallback = Callable[[PcmData], Coroutine[None, None, None]]
DisconnectCallback = Callable[[], Coroutine[None, None, None]]


def _task_done_callback(task: asyncio.Task[None]) -> None:
    if not task.cancelled() and task.exception() is not None:
        logger.error(
            "Background task %s failed", task.get_name(), exc_info=task.exception()
        )


class LiveAvatarRTCManager:
    """Joins HeyGen's avatar room and forwards remote video/audio to callbacks."""

    def __init__(
        self,
        on_video: VideoCallback,
        on_audio: AudioCallback,
        on_disconnect: DisconnectCallback,
    ) -> None:
        self._on_video = on_video
        self._on_audio = on_audio
        self._on_disconnect = on_disconnect

        self._room: rtc.Room | None = None
        self._connected = False
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self, room_url: str, token: str) -> None:
        room = rtc.Room()

        @room.on("participant_connected")
        def on_participant_connected(participant: rtc.RemoteParticipant) -> None:
            logger.info("LiveAvatar participant joined: %s", participant.identity)

        @room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                logger.info("Subscribed video track from %s", participant.identity)
                video_stream = rtc.VideoStream(track)
                self._create_task(self._consume_video(video_stream))
            elif track.kind == rtc.TrackKind.KIND_AUDIO:
                logger.info("Subscribed audio track from %s", participant.identity)
                audio_stream = rtc.AudioStream(track, sample_rate=48000, num_channels=2)
                self._create_task(self._consume_audio(audio_stream))

        @room.on("participant_disconnected")
        def on_participant_disconnected(participant: rtc.RemoteParticipant) -> None:
            logger.info(
                "LiveAvatar participant disconnected: %s; reason: %s",
                participant.identity,
                participant.disconnect_reason,
            )
            self._connected = False
            self._create_task(self._on_disconnect())
            if self._room is not None:
                self._create_task(self._room.disconnect())

        @room.on("disconnected")
        def on_disconnected(reason: str) -> None:
            # The "disconnected" callback may fire multiple times because we
            # also disconnect ourselves when the avatar leaves.
            if self._connected:
                logger.info("LiveAvatar room disconnected; reason: %s", reason)
                self._connected = False
                self._create_task(self._on_disconnect())

        logger.info("Connecting to LiveAvatar room url=%s", room_url)
        await room.connect(room_url, token)
        logger.info("Connected to LiveAvatar room")

        self._room = room
        self._connected = True

    async def close(self) -> None:
        try:
            await cancel_and_wait(*self._tasks)
            self._tasks.clear()
            if self._room is not None:
                await self._room.disconnect()
        finally:
            self._room = None
            self._connected = False

    async def _consume_video(self, video_stream: rtc.VideoStream) -> None:
        async for event in video_stream:
            lk_frame = event.frame.convert(rtc.VideoBufferType.RGBA)
            arr = np.frombuffer(lk_frame.data, dtype=np.uint8).reshape(
                lk_frame.height, lk_frame.width, 4
            )
            await self._on_video(av.VideoFrame.from_ndarray(arr, format="rgba"))

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

    def _create_task(self, coro: Coroutine[None, None, None]) -> None:
        task: asyncio.Task[None] = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(_task_done_callback)
