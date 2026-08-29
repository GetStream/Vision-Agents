import asyncio
import logging
from dataclasses import dataclass
from os import getenv
from typing import Any, Callable, Coroutine
from uuid import uuid4

import aiortc
import av
from getstream import AsyncStream
from getstream.models import CallRequest, MemberRequest
from getstream.video import rtc
from getstream.video.async_call import Call
from getstream.video.rtc.connection_manager import ConnectionManager
from getstream.video.rtc.pb.stream.video.sfu.event import events_pb2
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import (
    TrackType as StreamTrackType,
)
from getstream.video.rtc.track_util import FrameResampler, PcmData
from getstream.video.rtc.tracks import SubscriptionConfig, TrackSubscriptionConfig
from vision_agents.core.utils.utils import cancel_and_wait, get_vision_agents_version

from .track import AvatarInputTrack

logger = logging.getLogger(__name__)

_AVATAR_AUDIO_SAMPLE_RATE = 16000
_AVATAR_AUDIO_CHANNELS = 1
_CUSTOM_EVENT_END_UTTERANCE = "lemonslice.end_utterance"
_CUSTOM_EVENT_INTERRUPT = "lemonslice.interrupt"


@dataclass(frozen=True)
class StreamConnectionCredentials:
    """Credentials a LemonSlice avatar needs to join a Stream call."""

    api_key: str
    call_id: str
    call_type: str
    avatar_user_id: str
    avatar_token: str


class StreamRTCManager:
    """Stream-backed RTC manager for the LemonSlice avatar.

    Creates a Stream call, mints a call-scoped token for the avatar, joins as
    the plugin participant, publishes outgoing TTS audio, and dispatches
    incoming avatar audio and video to the supplied callbacks.

    flush() emits end-of-utterance RPC event to the avatar call.
    interrupt() emits the "interrupt" RPC event to stop the current avatar's utterance
    """

    def __init__(
        self,
        on_video: Callable[[av.VideoFrame], Coroutine[None, None, None]],
        on_audio: Callable[[PcmData], Coroutine[None, None, None]],
        on_disconnect: Callable[[], Coroutine[None, None, None]],
        stream_api_key: str | None = None,
        stream_api_secret: str | None = None,
        stream_call_type: str = "default",
        avatar_join_timeout: float = 30.0,
    ):
        """Create the RTC manager.

        Args:
            on_video: Async callback invoked for each avatar video frame.
            on_audio: Async callback invoked for each avatar audio chunk.
            on_disconnect: Async callback invoked when the avatar leaves or the call ends.
            stream_api_key: Stream API key. Uses STREAM_API_KEY env var if not provided.
            stream_api_secret: Stream API secret. Uses STREAM_API_SECRET env var if not provided.
            stream_call_type: Stream call type controlling the default feature set and
                per-role permissions for the call. The built-in "default" type is meant
                for 1:1/group video+audio calls: it enables audio, video, screensharing,
                recording, HLS broadcasting, transcription and ringing, and gives
                admins/hosts elevated permissions over regular participants.
                If you pass a custom call type, it must grant the `call_member` role
                the `join-call`, `read-call`, `send-audio`, and `send-video`
                capabilities — the plugin and avatar users are attached as members
                with that role so they can join regardless of the type's default
                user-role grants.
                See https://getstream.io/video/docs/api/call_types/builtin/ and
                https://getstream.io/video/docs/api/call_types/permissions/.
            avatar_join_timeout: Seconds to wait for the avatar participant to join
                the call before giving up.
        """
        stream_api_key = stream_api_key or getenv("STREAM_API_KEY")
        stream_api_secret = stream_api_secret or getenv("STREAM_API_SECRET")

        if not stream_api_key or not stream_api_secret:
            raise ValueError(
                "Stream API key and secret required. Set STREAM_API_KEY and "
                "STREAM_API_SECRET environment variables or pass them as parameters."
            )

        self._stream_api_key = stream_api_key
        self._stream_api_secret = stream_api_secret
        self._stream_call_type = stream_call_type
        self._avatar_join_timeout = avatar_join_timeout

        self._on_video = on_video
        self._on_audio = on_audio
        self._on_disconnect = on_disconnect

        version = get_vision_agents_version()
        client_kwargs: dict[str, Any] = {
            "api_key": self._stream_api_key,
            "api_secret": self._stream_api_secret,
            "user_agent": f"stream-vision-agents-{version}",
        }
        self._client = AsyncStream(**client_kwargs)

        self._plugin_user_id = f"plugin-{uuid4()}"
        self._avatar_user_id = f"avatar-{uuid4()}"

        self._call: Call | None = None
        self._connection: ConnectionManager | None = None
        self._input_track: AvatarInputTrack | None = None
        self._resampler = FrameResampler(
            rate=_AVATAR_AUDIO_SAMPLE_RATE, layout="mono", format="s16", frame_size=0
        )
        self._connected = False
        self._avatar_joined = asyncio.Event()
        self._event_id = 0
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def is_connected(self) -> bool:
        return self._connected

    def generate_credentials(self) -> StreamConnectionCredentials:
        call_id = f"lemonslice-{uuid4()}"
        call_cid = f"{self._stream_call_type}:{call_id}"
        avatar_token = self._client.create_call_token(
            self._avatar_user_id,
            call_cids=[call_cid],
            expiration=3600,
        )
        return StreamConnectionCredentials(
            api_key=self._stream_api_key,
            call_id=call_id,
            call_type=self._stream_call_type,
            avatar_user_id=self._avatar_user_id,
            avatar_token=avatar_token,
        )

    async def connect(self, credentials: StreamConnectionCredentials) -> None:
        """Join the Stream call and publish an outgoing audio track."""
        await self._client.create_user(
            id=self._plugin_user_id, name=self._plugin_user_id
        )
        await self._client.create_user(
            id=self._avatar_user_id, name=self._avatar_user_id
        )

        call = self._client.video.call(credentials.call_type, credentials.call_id)
        # Attach plugin + avatar users as members so they can join regardless of
        # the call type's per-role grants. See README for the contract on custom
        # call types.
        await call.get_or_create(
            data=CallRequest(
                created_by_id=self._plugin_user_id,
                members=[
                    MemberRequest(user_id=self._plugin_user_id, role="call_member"),
                    MemberRequest(user_id=self._avatar_user_id, role="call_member"),
                ],
            )
        )
        self._call = call

        subscription_config = SubscriptionConfig(
            default=TrackSubscriptionConfig(
                track_types=[
                    StreamTrackType.TRACK_TYPE_VIDEO,
                    StreamTrackType.TRACK_TYPE_AUDIO,
                ]
            )
        )

        connection = await rtc.join(
            call,
            self._plugin_user_id,
            subscription_config=subscription_config,
        )
        self._connection = connection

        input_track = AvatarInputTrack(
            sample_rate=_AVATAR_AUDIO_SAMPLE_RATE,
            channels=_AVATAR_AUDIO_CHANNELS,
        )
        self._input_track = input_track

        @connection.on("track_added")
        async def on_track_added(track_id: str, kind: str, user: Any) -> None:
            if user is None or user.user_id != self._avatar_user_id:
                return

            if kind == "video":
                logger.info("Received video track from LemonSlice avatar")
                track = connection.subscriber_pc.add_track_subscriber(track_id)
                if track is not None:
                    self._create_task(self._consume_video(track))

        @connection.on("audio")
        async def on_audio(pcm: PcmData) -> None:
            participant = pcm.participant
            if participant is None or participant.user_id != self._avatar_user_id:
                return
            await self._on_audio(pcm)

        @connection.on("participant_joined")
        async def on_participant_joined(event: events_pb2.ParticipantJoined) -> None:
            if event.participant.user_id != self._avatar_user_id:
                return
            logger.info("LemonSlice avatar joined the call")
            self._avatar_joined.set()

        @connection.on("participant_left")
        async def on_participant_left(event: events_pb2.ParticipantLeft) -> None:
            if event.participant.user_id != self._avatar_user_id:
                return
            logger.info("LemonSlice avatar left the call")
            self._connected = False
            self._create_task(self._on_disconnect())

        @connection.on("call_ended")
        async def on_call_ended(event: Any) -> None:
            if not self._connected:
                return
            logger.info("Stream call ended")
            self._connected = False
            self._create_task(self._on_disconnect())

        logger.info(
            f"Joining Stream call {credentials.call_type}:{credentials.call_id}"
        )
        await connection.__aenter__()
        await connection.add_tracks(audio=input_track)
        await connection.republish_tracks()
        self._connected = True
        logger.info("Connected to Stream call")

    async def wait_for_avatar(self) -> None:
        """Block until the avatar participant joins the call."""
        await asyncio.wait_for(
            self._avatar_joined.wait(), timeout=self._avatar_join_timeout
        )

    async def send_audio(self, pcm: PcmData) -> None:
        """Push a PCM chunk into the outgoing audio track."""
        if self._input_track is None or not self._connected:
            return
        await self._input_track.write(pcm)

    async def flush(self) -> None:
        """Signal end of a TTS segment to the avatar via a custom call event."""
        if self._call is None or not self._connected or self._input_track is None:
            return
        await self._input_track.write(
            PcmData(
                sample_rate=self._input_track.sample_rate,
                format=self._input_track.format,
                channels=self._input_track.channels,
            ),
            final=True,
        )
        pts = await self._input_track.pts()
        await self._call.send_call_event(
            user_id=self._plugin_user_id,
            custom={
                "type": _CUSTOM_EVENT_END_UTTERANCE,
                "pts": pts,
                "event_id": self._next_event_id(),
            },
        )

    async def interrupt(self) -> None:
        """Clear pending outgoing audio and signal the avatar to stop playback."""
        if self._input_track is not None:
            await self._input_track.flush()
        if self._call is None or not self._connected:
            return
        await self._call.send_call_event(
            user_id=self._plugin_user_id,
            custom={
                "type": _CUSTOM_EVENT_INTERRUPT,
                "event_id": self._next_event_id(),
            },
        )

    async def close(self) -> None:
        """Leave the Stream call and clean up resources."""
        try:
            await cancel_and_wait(*self._tasks)
            self._tasks.clear()

            if self._connection is not None:
                await self._connection.leave()

            if self._call is not None:
                await self._call.end()
            await self._client.aclose()
        finally:
            self._connection = None
            self._call = None
            self._input_track = None
            self._connected = False
            logger.debug("LemonSlice Stream RTC manager closed")

    async def _consume_video(self, track: aiortc.mediastreams.MediaStreamTrack) -> None:
        while True:
            frame = await track.recv()
            if isinstance(frame, av.VideoFrame):
                await self._on_video(frame)

    def _create_task(self, coro: Coroutine[None, None, None]) -> None:
        task: asyncio.Task[None] = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _next_event_id(self) -> int:
        self._event_id += 1
        return self._event_id
