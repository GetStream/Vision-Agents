"""Tencent TRTC EdgeTransport implementation.

Requires the liteav Python module from the Tencent RTC SDK to be built and
installed (see SDK python/python_x86_64). LD_LIBRARY_PATH must point to libliteav.so.
"""

import asyncio
import faulthandler
import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any, Optional

import TLSSigAPIv2
from dotenv import load_dotenv
from getstream.video.rtc import AudioStreamTrack
from getstream.video.rtc.track_util import AudioFormat, PcmData
from vision_agents.core.edge import Call, EdgeTransport, events
from vision_agents.core.edge.types import Connection, Participant, TrackType, User
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

_LITEAV_IMPORT_ERROR: Optional[ImportError] = None
try:
    from liteav import (
        AUDIO_CODEC_TYPE_PCM,
        AUDIO_OBTAIN_METHOD_CALLBACK,
        STREAM_TYPE_VIDEO_HIGH,
        AudioEncodeParams,
        CreateTRTCCloud,
        DestroyTRTCCloud,
        EnterRoomParams,
        TrtcString,
        TRTCCloudDelegate,
        TRTC_ROLE_ANCHOR,
        TRTC_SCENE_RECORD,
        VideoEncodeParams,
        cdata,
    )
except ImportError as e:
    _LITEAV_IMPORT_ERROR = e
    CreateTRTCCloud = None
    DestroyTRTCCloud = None
    TRTCCloudDelegate = None  # type: ignore[misc, assignment]
else:
    _LITEAV_IMPORT_ERROR = None

try:
    from liteav import TRTC_SCENE_CALL, TRTC_SCENE_VIDEOCALL
except ImportError:
    TRTC_SCENE_VIDEOCALL = None  # type: ignore[assignment]
    TRTC_SCENE_CALL = None  # type: ignore[assignment]


def _require_liteav() -> None:
    if _LITEAV_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Tencent TRTC edge requires the liteav Python module from the Tencent RTC SDK. "
            "Build it from the SDK's python/python_x86_64 directory (./build.sh) and set "
            "LD_LIBRARY_PATH to the directory containing libliteav.so."
        ) from _LITEAV_IMPORT_ERROR


def _resolve_room_scene() -> tuple[int, str]:
    configured_scene = os.getenv("TENCENT_TRTC_SCENE", "auto").strip().lower()
    scene_map = {
        "record": TRTC_SCENE_RECORD,
        "videocall": TRTC_SCENE_VIDEOCALL,
        "call": TRTC_SCENE_CALL,
    }
    if configured_scene == "auto":
        for candidate in ("videocall", "call"):
            scene = scene_map[candidate]
            if scene is not None:
                return scene, candidate
        return TRTC_SCENE_RECORD, "record"

    if configured_scene in scene_map:
        scene = scene_map[configured_scene]
        if scene is None:
            raise ValueError(
                f"TENCENT_TRTC_SCENE={configured_scene} is not supported by this liteav binding."
            )
        return scene, configured_scene

    raise ValueError(
        "TENCENT_TRTC_SCENE must be one of: auto, videocall, call, record."
    )


def _extract_pcm(frame: Any) -> bytes:
    """Extract PCM bytes from a SWIG AudioFrame using the GIL-safe cdata path.

    AudioFrame_getdata / PcmData call PyBytes_FromStringAndSize inside
    a SWIG_PYTHON_THREAD_BEGIN_ALLOW block (without the GIL).  Under
    multi-threading this corrupts CPython's allocator and segfaults.

    cdata() avoids the issue: it copies the raw pointer into a C struct
    without the GIL, then creates the Python bytes object *with* the GIL.
    """
    sz = frame.size()
    if sz <= 0:
        return b""
    raw = cdata(frame.data(), sz)
    if isinstance(raw, bytes):
        return raw
    return raw.encode("utf-8", "surrogateescape")


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
        _require_liteav()
        super().__init__()

        if sdk_app_id is None:
            raw = os.environ.get("TENCENT_SDKAppID")
            if raw is None:
                raise ValueError(
                    "sdk_app_id must be provided or set TENCENT_SDKAppID env var"
                )
            sdk_app_id = int(raw)

        if not key:
            key = os.environ.get("TENCENT_SDKSecretKey")

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
        self._room_scene, self._room_scene_name = _resolve_room_scene()

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
        if self._delegate is not None:
            self._delegate.__disown__()
            self._delegate = None
        self._connection = None
        self._call = None

    def open_demo(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def join(
        self, agent: "Agent", call: TencentCall, **kwargs: Any
    ) -> Connection:
        _require_liteav()
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

        if _TencentDelegate is None:
            _require_liteav()
        assert _TencentDelegate is not None
        delegate = _TencentDelegate(
            edge=self,
            connection=self._connection,
            loop=self._loop,
        )
        self._delegate = delegate
        self._cloud = CreateTRTCCloud(delegate)

        room_param = EnterRoomParams()
        room_param.room.sdk_app_id = self._sdk_app_id
        if call.str_room_id is not None:
            room_param.room.str_room_id = TrtcString(call.str_room_id)
        else:
            room_param.room.room_id = call.room_id
        room_param.room.user_id = TrtcString(self._agent_user_id or "")
        room_param.room.user_sig = TrtcString(user_sig)
        room_param.role = TRTC_ROLE_ANCHOR
        room_param.scene = self._room_scene
        room_param.use_pixel_frame_input = True
        room_param.use_pixel_frame_output = True
        room_param.audio_obtain_params.audio_obtain_method = (
            AUDIO_OBTAIN_METHOD_CALLBACK
        )
        room_param.audio_obtain_params.output_sample_rate = SAMPLE_RATE
        room_param.audio_obtain_params.output_channles = CHANNELS
        room_param.audio_obtain_params.output_frame_length_ms = FRAME_MS
        room_param.audio_obtain_params.output_audio_codec_type = AUDIO_CODEC_TYPE_PCM

        assert self._cloud is not None
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
    ) -> None:
        pass

    def add_track_subscriber(self, track_id: str) -> Optional[Any]:
        return self._incoming_video_tracks.get(track_id)

    async def send_custom_event(self, data: dict[str, Any]) -> None:
        if self._cloud is None:
            return
        raw = str(data).encode("utf-8")[: 5 * 1024]
        self._cloud.SendCustomCmdMsg(1, bytearray(raw), True, True)

    def _emit_audio_received(self, user_id: str, pcm_bytes: bytes) -> None:
        if not self._loop or not pcm_bytes:
            return
        self._loop.call_soon_threadsafe(
            self._process_audio_on_loop, user_id, pcm_bytes
        )

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
        track = TencentIncomingVideoTrack(loop=self._loop)
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


_TencentDelegate: Any = None
if TRTCCloudDelegate is not None:

    class _TencentDelegateCls(TRTCCloudDelegate):
        """TRTCCloudDelegate that forwards callbacks to the edge and connection.

        All callbacks are wrapped in try/except because any unhandled Python
        exception in a SWIG director callback triggers Swig::DirectorMethodException
        on the C++ side, which calls std::terminate and kills the process.
        """

        def __init__(
            self,
            edge: TencentEdge,
            connection: TencentConnection,
            loop: asyncio.AbstractEventLoop,
        ):
            TRTCCloudDelegate.__init__(self)
            self._edge = edge
            self._connection = connection
            self._agent_user_id = edge._agent_user_id or ""

        def OnError(self, error: int) -> None:
            try:
                logger.error("Tencent TRTC OnError: %s", error)
                self._edge._emit_call_ended()
            except BaseException:
                logger.exception("OnError callback failed")

        def OnEnterRoom(self) -> None:
            try:
                logger.info("Tencent TRTC OnEnterRoom")
                cloud = self._edge._cloud
                if cloud is None:
                    return
                param = AudioEncodeParams()
                param.sample_rate = SAMPLE_RATE
                param.channels = CHANNELS
                param.bitrate_bps = 54000
                cloud.CreateLocalAudioChannel(param)

                if self._edge._outgoing_video_track is not None:
                    cloud.CreateLocalVideoChannel(STREAM_TYPE_VIDEO_HIGH)
            except BaseException:
                logger.exception("OnEnterRoom callback failed")

        def OnExitRoom(self) -> None:
            try:
                logger.info("Tencent TRTC OnExitRoom")
                self._edge._emit_call_ended()
            except BaseException:
                logger.exception("OnExitRoom callback failed")

        def OnLocalAudioChannelCreated(self) -> None:
            try:
                if self._edge._audio_track and self._edge._cloud:
                    self._edge._audio_track.set_cloud(self._edge._cloud)
            except BaseException:
                logger.exception("OnLocalAudioChannelCreated callback failed")

        def OnConnectionStateChanged(self, old_state: int, new_state: int) -> None:
            logger.debug(
                "Tencent TRTC connection state: %s -> %s", old_state, new_state
            )

        def OnLocalAudioChannelDestroyed(self) -> None:
            pass

        def OnLocalVideoChannelCreated(self, stream_type: int) -> None:
            try:
                logger.info(
                    "Tencent TRTC OnLocalVideoChannelCreated: stream_type=%s",
                    stream_type,
                )
                edge = self._edge
                if edge._outgoing_video_track and edge._cloud and edge._loop:
                    vp = VideoEncodeParams()
                    vp.frame_rate = edge._video_fps
                    vp.bitrate_bps = 1_000_000
                    vp.gop_in_seconds = 3
                    edge._cloud.SetVideoEncodeParam(STREAM_TYPE_VIDEO_HIGH, vp)
                    edge._outgoing_video_track.set_cloud(edge._cloud, edge._loop)
            except BaseException:
                logger.exception("OnLocalVideoChannelCreated callback failed")

        def OnLocalVideoChannelDestroyed(self, stream_type: int) -> None:
            pass

        def OnRequestChangeVideoEncodeBitrate(
            self, stream_type: int, bitrate_bps: int
        ) -> None:
            pass

        def OnRequestKeyFrame(self, stream_type: int) -> None:
            logger.debug("Tencent TRTC OnRequestKeyFrame: stream_type=%s", stream_type)

        def OnRemoteAudioAvailable(self, user_id: str, available: bool) -> None:
            try:
                if user_id and user_id != self._agent_user_id:
                    if available:
                        self._edge._emit_track_added(user_id)
                    else:
                        self._edge._emit_track_removed(user_id)
            except BaseException:
                logger.exception("OnRemoteAudioAvailable callback failed")

        def OnRemoteVideoAvailable(
            self, user_id: str, available: bool, stream_type: int
        ) -> None:
            try:
                if not user_id or user_id == self._agent_user_id:
                    return
                if available:
                    logger.info(
                        "Tencent TRTC video available from %s (stream_type=%s)",
                        user_id,
                        stream_type,
                    )
                    self._edge._emit_video_track_added(user_id)
                else:
                    logger.info("Tencent TRTC video unavailable from %s", user_id)
                    self._edge._emit_video_track_removed(user_id)
            except BaseException:
                logger.exception("OnRemoteVideoAvailable callback failed")

        def OnRemoteVideoFrameReceived(
            self, user_id: str, stream_type: int, frame: Any
        ) -> None:
            pass

        def OnRemotePixelFrameReceived(
            self, user_id: str, stream_type: int, frame: Any
        ) -> None:
            try:
                if not user_id or user_id == self._agent_user_id:
                    return
                width = frame.width
                height = frame.height
                if width <= 0 or height <= 0:
                    return
                sz = frame.size()
                if sz <= 0:
                    return
                raw = cdata(frame.data(), sz)
                if isinstance(raw, bytes):
                    yuv_bytes = raw
                else:
                    yuv_bytes = raw.encode("utf-8", "surrogateescape")
                self._edge._push_video_frame(
                    user_id, yuv_bytes, width, height, frame.pts
                )
            except BaseException:
                logger.exception("OnRemotePixelFrameReceived callback failed")

        def OnSeiMessageReceived(
            self, user_id: str, stream_type: int, message_type: int, message: Any
        ) -> None:
            pass

        def OnReceiveCustomCmdMsg(
            self, user_id: str, cmd_id: int, seq: int, message: Any
        ) -> None:
            pass

        def OnMissCustomCmdMsg(
            self, user_id: str, cmd_id: int, error_code: int, missed: int
        ) -> None:
            pass

        def OnNetworkQuality(self, local_quality: Any, remote_qualities: Any) -> None:
            pass

        def OnRemoteUserEnterRoom(self, info: Any) -> None:
            try:
                user_id = info.user_id.GetValue() if info and info.user_id else ""
                if user_id and user_id != self._agent_user_id:
                    self._connection._on_remote_entered()
                logger.info("Tencent TRTC OnRemoteUserEnterRoom: %s", user_id)
            except BaseException:
                logger.exception("OnRemoteUserEnterRoom callback failed")

        def OnRemoteUserExitRoom(self, info: Any, reason: int) -> None:
            try:
                user_id = info.user_id.GetValue() if info and info.user_id else ""
                if user_id and user_id != self._agent_user_id:
                    self._connection._on_remote_left()
                logger.info(
                    "Tencent TRTC OnRemoteUserExitRoom: %s reason=%s", user_id, reason
                )
            except BaseException:
                logger.exception("OnRemoteUserExitRoom callback failed")

        def OnRemoteAudioReceived(self, user_id: str, frame: Any) -> None:
            try:
                if not user_id or user_id == self._agent_user_id:
                    return
                data = _extract_pcm(frame)
                if data:
                    self._edge._emit_audio_received(user_id, data)
            except BaseException:
                logger.exception("OnRemoteAudioReceived failed")

        def OnRemoteMixedAudioReceived(self, frame: Any) -> None:
            # Skip mixed audio — per-user callbacks already deliver individual
            # streams and processing both doubles GIL / buffer pressure with
            # no benefit for single-speaker scenarios.
            pass

    _TencentDelegate = _TencentDelegateCls
