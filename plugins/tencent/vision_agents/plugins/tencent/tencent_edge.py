"""Tencent TRTC EdgeTransport implementation.

Requires the `liteav` Python package from PyPI, which only ships manylinux
wheels (x86_64 / aarch64). On macOS/Windows the import will fail and
``tencent.Edge()`` raises at construction time.
"""

import asyncio
import faulthandler
import json
import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any, Optional

import TLSSigAPIv2
from dotenv import load_dotenv
from getstream.video.rtc import AudioStreamTrack
from getstream.video.rtc.track_util import AudioFormat, PcmData
from vision_agents.core.agents.conversation import Conversation, InMemoryConversation
from vision_agents.core.edge import Call, EdgeTransport, events
from vision_agents.core.edge.types import Connection, Participant, TrackType, User
from vision_agents.plugins.tencent.bindings import (
    AUDIO_CODEC_TYPE_PCM,
    AUDIO_OBTAIN_METHOD_CALLBACK,
    STREAM_TYPE_VIDEO_HIGH,
    TRTC_ROLE_ANCHOR,
    CreateTRTCCloud,
    DestroyTRTCCloud,
    EnterRoomParams,
    TrtcString,
    require_liteav,
)
from vision_agents.plugins.tencent.delegate import TencentDelegate
from vision_agents.plugins.tencent.scene import resolve_room_scene
from vision_agents.plugins.tencent.tracks import (
    CHANNELS,
    FRAME_MS,
    SAMPLE_RATE,
    TencentAudioTrack,
    TencentIncomingVideoTrack,
    TencentOutgoingVideoTrack,
    _DEFAULT_VIDEO_FPS,
)

if TYPE_CHECKING:
    from vision_agents.core import Agent

faulthandler.enable()
load_dotenv()

logger = logging.getLogger(__name__)


class TencentCall(Call):
    """Call backed by a Tencent TRTC room."""

    def __init__(
        self,
        call_id: str,
        room_id: int = 0,
        str_room_id: str | None = None,
    ):
        self._id = call_id
        self.room_id = room_id
        self.str_room_id = str_room_id

    @property
    def id(self) -> str:
        return self._id


class TencentConnection(Connection):
    """Connection to a Tencent TRTC room."""

    def __init__(
        self,
        agent_user_id: str,
        loop: asyncio.AbstractEventLoop,
        edge: "TencentEdge",
    ):
        super().__init__()
        self._agent_user_id = agent_user_id
        self._loop = loop
        self._edge = edge
        self._participant_joined = asyncio.Event()
        self._idle_since: float = 0.0
        self._remote_count = 0
        self._lock = threading.Lock()

    def idle_since(self) -> float:
        return self._idle_since

    async def wait_for_participant(self, timeout: Optional[float] = None) -> None:
        await asyncio.wait_for(self._participant_joined.wait(), timeout=timeout)

    async def close(self) -> None:
        self._edge._exit_room()

    def _on_remote_entered(self) -> None:
        with self._lock:
            self._remote_count += 1
            self._idle_since = 0.0
        self._loop.call_soon_threadsafe(self._participant_joined.set)

    def _on_remote_left(self) -> None:
        with self._lock:
            self._remote_count -= 1
            if self._remote_count <= 0 and self._idle_since == 0.0:
                self._idle_since = time.time()


class TencentEdge(EdgeTransport[TencentCall]):
    """Edge transport using Tencent TRTC."""

    def __init__(
        self,
        sdk_app_id: int | None = None,
        user_sig: str | None = None,
        key: str | None = None,
        video_fps: int = _DEFAULT_VIDEO_FPS,
    ):
        require_liteav()
        super().__init__()

        if sdk_app_id is None:
            raw = os.environ.get("TENCENT_SDK_APP_ID")
            if raw is None:
                raise ValueError(
                    "sdk_app_id must be provided or set TENCENT_SDK_APP_ID env var"
                )
            sdk_app_id = int(raw)

        if not key:
            key = os.environ.get("TENCENT_SDK_SECRET_KEY")

        self._sdk_app_id = sdk_app_id
        self._user_sig = user_sig
        self._key = key
        self._video_fps = video_fps
        self._cloud = None
        self._connection: Optional[TencentConnection] = None
        self._call: Optional[TencentCall] = None
        self._agent_user_id: Optional[str] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._audio_track: Optional[TencentAudioTrack] = None
        self._outgoing_video_track: Optional[TencentOutgoingVideoTrack] = None
        self._incoming_video_tracks: dict[str, TencentIncomingVideoTrack] = {}
        self._delegate = None
        self._room_scene, self._room_scene_name = resolve_room_scene()

    async def authenticate(self, user: User) -> None:
        self._agent_user_id = user.id

    async def create_call(
        self, call_id: str, room_id: Optional[int] = None, **kwargs: Any
    ) -> TencentCall:
        if room_id is not None:
            return TencentCall(call_id=call_id, room_id=room_id)
        try:
            return TencentCall(call_id=call_id, room_id=int(call_id))
        except ValueError:
            return TencentCall(call_id=call_id, str_room_id=call_id)

    def create_audio_track(self) -> AudioStreamTrack:
        track = TencentAudioTrack()
        self._audio_track = track
        return track  # type: ignore[return-value]

    def _exit_room(self) -> None:
        if self._cloud is not None:
            self._cloud.ExitRoom()
        if self._audio_track is not None:
            self._audio_track.stop()
        if self._outgoing_video_track is not None:
            self._outgoing_video_track.stop()
        for track in self._incoming_video_tracks.values():
            track.stop()

    async def close(self) -> None:
        self._exit_room()
        if self._cloud is not None:
            DestroyTRTCCloud(self._cloud)
            self._cloud = None
        self._audio_track = None
        self._outgoing_video_track = None
        self._incoming_video_tracks.clear()
        self._delegate = None
        self._connection = None
        self._call = None

    def open_demo(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def join(
        self, agent: "Agent", call: TencentCall, **kwargs: Any
    ) -> Connection:
        require_liteav()
        user_sig = self._user_sig
        if not user_sig and self._key and self._agent_user_id:
            api = TLSSigAPIv2.TLSSigAPIv2(self._sdk_app_id, self._key)
            user_sig = api.gen_sig(self._agent_user_id)
        if not user_sig:
            raise ValueError(
                "Tencent edge requires user_sig (or key= with TLSSigAPIv2)"
            )

        self._call = call
        self._loop = asyncio.get_running_loop()
        self._connection = TencentConnection(
            agent_user_id=agent.agent_user.id or "",
            loop=self._loop,
            edge=self,
        )

        if TencentDelegate is None:
            # require_liteav() above should have raised, so this is
            # defensive only — keeps the type checker honest and
            # survives `python -O` (which strips asserts).
            raise RuntimeError("liteav delegate is not available")
        delegate = TencentDelegate(
            edge=self,
            connection=self._connection,
            loop=self._loop,
        )
        self._delegate = delegate
        self._cloud = CreateTRTCCloud(delegate)

        room_param = EnterRoomParams()
        room_param.room.sdk_app_id = self._sdk_app_id
        if call.str_room_id is not None:
            room_param.room.str_room_id = TrtcString(call.str_room_id)  # type: ignore[call-arg]
        else:
            room_param.room.room_id = call.room_id
        room_param.room.user_id = TrtcString(self._agent_user_id or "")  # type: ignore[call-arg]
        room_param.room.user_sig = TrtcString(user_sig)  # type: ignore[call-arg]
        room_param.role = TRTC_ROLE_ANCHOR
        room_param.scene = self._room_scene  # type: ignore[assignment]
        room_param.use_pixel_frame_input = True
        room_param.use_pixel_frame_output = True
        room_param.audio_obtain_params.audio_obtain_method = (
            AUDIO_OBTAIN_METHOD_CALLBACK
        )
        room_param.audio_obtain_params.output_sample_rate = SAMPLE_RATE
        room_param.audio_obtain_params.output_channels = CHANNELS
        room_param.audio_obtain_params.output_frame_length_ms = FRAME_MS
        room_param.audio_obtain_params.output_audio_codec_type = AUDIO_CODEC_TYPE_PCM

        if self._cloud is None:
            # CreateTRTCCloud was just called above; this would only
            # trip if the SDK returned a null pointer.
            raise RuntimeError("CreateTRTCCloud returned no cloud handle")
        logger.info("Tencent TRTC scene selected: %s", self._room_scene_name)
        self._cloud.EnterRoom(room_param)
        return self._connection

    async def publish_tracks(
        self,
        audio_track: Optional[Any] = None,
        video_track: Optional[Any] = None,
    ) -> None:
        if video_track is not None:
            self._outgoing_video_track = TencentOutgoingVideoTrack(
                source=video_track, fps=self._video_fps
            )
            if self._cloud is not None and self._loop is not None:
                self._cloud.CreateLocalVideoChannel(STREAM_TYPE_VIDEO_HIGH)

    async def create_conversation(
        self, call: Call, user: User, instructions: str
    ) -> Conversation:
        return InMemoryConversation(messages=[], instructions=instructions)

    def add_track_subscriber(self, track_id: str) -> Optional[Any]:
        return self._incoming_video_tracks.get(track_id)

    async def send_custom_event(self, data: dict[str, Any]) -> None:
        if self._cloud is None:
            return
        # TRTC peers across other SDKs (JS, mobile) expect JSON, not the
        # Python `repr` of a dict. Truncate to the SDK's 5 KiB cap.
        raw = json.dumps(data).encode("utf-8")[: 5 * 1024]
        self._cloud.SendCustomCmdMsg(1, bytearray(raw), True, True)

    def _emit_audio_received(self, user_id: str, pcm_bytes: bytes) -> None:
        if not self._loop or not pcm_bytes:
            return
        self._loop.call_soon_threadsafe(self._process_audio_on_loop, user_id, pcm_bytes)

    def _process_audio_on_loop(self, user_id: str, pcm_bytes: bytes) -> None:
        pcm = PcmData.from_bytes(
            pcm_bytes,
            sample_rate=SAMPLE_RATE,
            channels=CHANNELS,
            format=AudioFormat.S16,
        )
        participant = Participant(original=None, user_id=user_id, id=user_id)
        event = events.AudioReceivedEvent(
            plugin_name="tencent",
            pcm_data=pcm,
            participant=participant,
        )
        self.events.send(event)

    def _emit_track_added(self, user_id: str) -> None:
        if self._loop:
            event = events.TrackAddedEvent(
                plugin_name="tencent",
                track_id=f"{user_id}-audio",
                track_type=TrackType.AUDIO,
                participant=Participant(original=None, user_id=user_id, id=user_id),
            )
            self._loop.call_soon_threadsafe(self.events.send, event)

    def _emit_track_removed(self, user_id: str) -> None:
        if self._loop:
            event = events.TrackRemovedEvent(
                plugin_name="tencent",
                track_id=f"{user_id}-audio",
                track_type=TrackType.AUDIO,
                participant=Participant(original=None, user_id=user_id, id=user_id),
            )
            self._loop.call_soon_threadsafe(self.events.send, event)

    def _emit_call_ended(self) -> None:
        if self._loop:
            self._loop.call_soon_threadsafe(
                self.events.send,
                events.CallEndedEvent(plugin_name="tencent"),
            )

    def _emit_video_track_added(self, user_id: str) -> None:
        if not self._loop:
            return
        track_id = f"{user_id}-video"
        track = TencentIncomingVideoTrack()
        self._incoming_video_tracks[track_id] = track
        event = events.TrackAddedEvent(
            plugin_name="tencent",
            track_id=track_id,
            track_type=TrackType.VIDEO,
            participant=Participant(original=None, user_id=user_id, id=user_id),
        )
        self._loop.call_soon_threadsafe(self.events.send, event)

    def _emit_video_track_removed(self, user_id: str) -> None:
        if not self._loop:
            return
        track_id = f"{user_id}-video"
        track = self._incoming_video_tracks.pop(track_id, None)
        if track is not None:
            track.stop()
        event = events.TrackRemovedEvent(
            plugin_name="tencent",
            track_id=track_id,
            track_type=TrackType.VIDEO,
            participant=Participant(original=None, user_id=user_id, id=user_id),
        )
        self._loop.call_soon_threadsafe(self.events.send, event)

    def _push_video_frame(
        self, user_id: str, yuv_bytes: bytes, width: int, height: int, pts: int
    ) -> None:
        track_id = f"{user_id}-video"
        track = self._incoming_video_tracks.get(track_id)
        if track is None:
            return
        track.push_frame(yuv_bytes, width, height, pts)
