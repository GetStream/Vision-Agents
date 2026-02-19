"""Tencent TRTC EdgeTransport implementation.

Requires the liteav Python module from the Tencent RTC SDK to be built and
installed (see SDK python/python_x86_64). LD_LIBRARY_PATH must point to libliteav.so.
"""

import asyncio
import logging
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Optional

from getstream.video.rtc import AudioStreamTrack
from getstream.video.rtc.track_util import AudioFormat, PcmData
from vision_agents.core.edge import Call, EdgeTransport, events
from vision_agents.core.edge.types import Connection, Participant, TrackType, User

if TYPE_CHECKING:
    from vision_agents.core import Agent

logger = logging.getLogger(__name__)

_LITEAV_IMPORT_ERROR: Optional[ImportError] = None
try:
    from liteav import (
        AUDIO_CODEC_TYPE_PCM,
        AudioEncodeParams,
        AudioFrame,
        CreateTRTCCloud,
        DestroyTRTCCloud,
        EnterRoomParams,
        TrtcString,
        TRTCCloudDelegate,
        TRTC_ROLE_ANCHOR,
        TRTC_SCENE_RECORD,
    )
except ImportError as e:
    _LITEAV_IMPORT_ERROR = e
    CreateTRTCCloud = None
    DestroyTRTCCloud = None
    TRTCCloudDelegate = None  # type: ignore[misc, assignment]
else:
    _LITEAV_IMPORT_ERROR = None


def _require_liteav() -> None:
    if _LITEAV_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Tencent TRTC edge requires the liteav Python module from the Tencent RTC SDK. "
            "Build it from the SDK's python/python_x86_64 directory (./build.sh) and set "
            "LD_LIBRARY_PATH to the directory containing libliteav.so."
        ) from _LITEAV_IMPORT_ERROR


FRAME_MS = 20
SAMPLE_RATE = 16000
CHANNELS = 1
BYTES_PER_20MS = 640


class TencentCall(Call):
    """Call backed by a Tencent TRTC room."""

    def __init__(self, call_id: str, room_id: int):
        self._id = call_id
        self.room_id = room_id

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


class TencentAudioTrack:
    """Duck-typed audio track: write(PcmData), stop(), flush(). Sends via Tencent SendAudioFrame."""

    def __init__(self, edge: "TencentEdge"):
        self._edge = edge
        self._queue: deque[bytes] = deque()
        self._lock = threading.Lock()
        self._cloud = None
        self._running = True
        self._sender_thread: Optional[threading.Thread] = None
        self._pts = 10

    def set_cloud(self, cloud: Any) -> None:
        self._cloud = cloud
        if self._sender_thread is None:
            self._sender_thread = threading.Thread(target=self._send_loop, daemon=True)
            self._sender_thread.start()

    def write(self, pcm: PcmData) -> None:
        if not self._running or pcm is None:
            return
        try:
            if pcm.samples is not None and pcm.samples.size > 0:
                data = pcm.samples.tobytes()
            else:
                data = pcm.to_bytes()
            if data:
                with self._lock:
                    self._queue.append(data)
        except (AttributeError, TypeError) as e:
            logger.debug("TencentAudioTrack.write: %s", e)
        except Exception as e:
            logger.exception("TencentAudioTrack.write failed: %s", e)

    def stop(self) -> None:
        self._running = False

    async def flush(self) -> None:
        with self._lock:
            self._queue.clear()

    def _send_loop(self) -> None:
        while self._running and self._cloud is not None:
            chunk = None
            with self._lock:
                if self._queue:
                    chunk = self._queue.popleft()
            if chunk:
                offset = 0
                while offset < len(chunk) and self._running:
                    frame_bytes = chunk[offset : offset + BYTES_PER_20MS]
                    offset += len(frame_bytes)
                    if len(frame_bytes) < BYTES_PER_20MS and len(frame_bytes) > 0:
                        frame_bytes = frame_bytes + b"\x00" * (
                            BYTES_PER_20MS - len(frame_bytes)
                        )
                    if len(frame_bytes) == BYTES_PER_20MS:
                        try:
                            frame = AudioFrame()
                            frame.sample_rate = SAMPLE_RATE
                            frame.channels = CHANNELS
                            frame.bits_per_sample = 16
                            frame.codec = AUDIO_CODEC_TYPE_PCM
                            frame.pts = self._pts
                            self._pts += FRAME_MS
                            frame.SetData(frame_bytes)
                            self._cloud.SendAudioFrame(frame)
                        except Exception as e:
                            logger.debug("SendAudioFrame: %s", e)
            time.sleep(0.01)


class TencentEdge(EdgeTransport[TencentCall]):
    """Edge transport using Tencent TRTC. Plugin-only; no core changes."""

    def __init__(
        self,
        sdk_app_id: int,
        user_sig: Optional[str] = None,
        key: Optional[str] = None,
    ):
        _require_liteav()
        super().__init__()
        self._sdk_app_id = sdk_app_id
        self._user_sig = user_sig
        self._key = key
        self._cloud = None
        self._connection: Optional[TencentConnection] = None
        self._call: Optional[TencentCall] = None
        self._agent_user_id: Optional[str] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._audio_track: Optional[TencentAudioTrack] = None
        self._delegate = None

    async def create_user(self, user: User) -> None:
        self._agent_user_id = user.id

    async def create_call(
        self, call_id: str, room_id: Optional[int] = None, **kwargs: Any
    ) -> TencentCall:
        rid = room_id if room_id is not None else int(call_id)
        return TencentCall(call_id=call_id, room_id=rid)

    def create_audio_track(self) -> AudioStreamTrack:
        track = TencentAudioTrack(edge=self)
        self._audio_track = track
        return track  # type: ignore[return-value]

    def _exit_room(self) -> None:
        if self._cloud is not None:
            try:
                self._cloud.ExitRoom()
            except Exception as e:
                logger.debug("ExitRoom: %s", e)
        if self._audio_track is not None:
            self._audio_track.stop()

    async def close(self) -> None:
        self._exit_room()
        if self._cloud is not None:
            try:
                DestroyTRTCCloud(self._cloud)
            except Exception as e:
                logger.debug("DestroyTRTCCloud: %s", e)
            self._cloud = None
        if self._audio_track is not None:
            self._audio_track = None
        if self._delegate is not None:
            try:
                self._delegate.__disown__()
            except Exception:
                pass
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
            try:
                import TLSSigAPIv2

                api = TLSSigAPIv2.TLSSigAPIv2(self._sdk_app_id, self._key)
                user_sig = api.gen_sig(self._agent_user_id)
            except ImportError:
                raise RuntimeError(
                    "Provide user_sig or install TLSSigAPIv2 and pass key= to generate it"
                )
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
        room_param.room.room_id = call.room_id
        room_param.room.user_id = TrtcString(self._agent_user_id or "")
        room_param.room.user_sig = TrtcString(user_sig)
        room_param.role = TRTC_ROLE_ANCHOR
        room_param.scene = TRTC_SCENE_RECORD
        room_param.audio_obtain_params.audio_obtain_method = 1
        room_param.audio_obtain_params.output_sample_rate = SAMPLE_RATE
        room_param.audio_obtain_params.output_channles = CHANNELS
        room_param.audio_obtain_params.output_frame_length_ms = FRAME_MS
        room_param.audio_obtain_params.output_audio_codec_type = 0

        assert self._cloud is not None
        self._cloud.EnterRoom(room_param)
        return self._connection

    async def publish_tracks(
        self,
        audio_track: Optional[Any] = None,
        video_track: Optional[Any] = None,
    ) -> None:
        pass

    async def create_conversation(
        self, call: Call, user: User, instructions: str
    ) -> None:
        pass

    def add_track_subscriber(self, track_id: str) -> Optional[Any]:
        return None

    async def send_custom_event(self, data: dict[str, Any]) -> None:
        if self._cloud is None:
            return
        try:
            raw = str(data).encode("utf-8")[: 5 * 1024]
            self._cloud.SendCustomCmdMsg(1, bytearray(raw), True, True)
        except Exception as e:
            logger.debug("SendCustomCmdMsg: %s", e)

    def _emit_audio_received(
        self, user_id: str, pcm_bytes: bytes, sample_rate: int, channels: int
    ) -> None:
        if not self._loop or not pcm_bytes:
            return
        try:
            pcm = PcmData.from_bytes(
                pcm_bytes,
                sample_rate=sample_rate,
                channels=channels,
                format=AudioFormat.S16,
            )
        except Exception as e:
            logger.debug("PcmData.from_bytes: %s", e)
            return
        participant = Participant(original=None, user_id=user_id, id=user_id)
        event = events.AudioReceivedEvent(
            plugin_name="tencent",
            pcm_data=pcm,
            participant=participant,
        )
        self._loop.call_soon_threadsafe(self.events.send, event)

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


_TencentDelegate: Any = None
if TRTCCloudDelegate is not None:

    class _TencentDelegateCls(TRTCCloudDelegate):
        """TRTCCloudDelegate that forwards callbacks to the edge and connection."""

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
            logger.error("Tencent TRTC OnError: %s", error)
            self._edge._emit_call_ended()

        def OnEnterRoom(self) -> None:
            logger.info("Tencent TRTC OnEnterRoom")
            cloud = self._edge._cloud
            if cloud is None:
                return
            param = AudioEncodeParams()
            param.sample_rate = SAMPLE_RATE
            param.channels = CHANNELS
            param.bitrate_bps = 54000
            cloud.CreateLocalAudioChannel(param)

        def OnExitRoom(self) -> None:
            logger.info("Tencent TRTC OnExitRoom")
            cloud = self._edge._cloud
            if cloud is not None:
                cloud.DestroyLocalAudioChannel()
            self._edge._emit_call_ended()

        def OnLocalAudioChannelCreated(self) -> None:
            if self._edge._audio_track and self._edge._cloud:
                self._edge._audio_track.set_cloud(self._edge._cloud)

        def OnLocalAudioChannelDestroyed(self) -> None:
            pass

        def OnRemoteUserEnterRoom(self, info: Any) -> None:
            user_id = info.user_id.GetValue() if info and info.user_id else ""
            if user_id and user_id != self._agent_user_id:
                self._connection._on_remote_entered()
                self._edge._emit_track_added(user_id)
            logger.info("Tencent TRTC OnRemoteUserEnterRoom: %s", user_id)

        def OnRemoteUserExitRoom(self, info: Any, reason: int) -> None:
            user_id = info.user_id.GetValue() if info and info.user_id else ""
            if user_id and user_id != self._agent_user_id:
                self._connection._on_remote_left()
                self._edge._emit_track_removed(user_id)
            logger.info(
                "Tencent TRTC OnRemoteUserExitRoom: %s reason=%s", user_id, reason
            )

        def OnRemoteAudioReceived(self, user_id: str, frame: Any) -> None:
            if not frame:
                return
            try:
                data = frame.PcmData()
                sr = getattr(frame, "sample_rate", None) or SAMPLE_RATE
                ch = getattr(frame, "channels", None) or CHANNELS
                if data:
                    self._edge._emit_audio_received(user_id, data, sr, ch)
            except Exception as e:
                logger.debug("OnRemoteAudioReceived: %s", e)

        def OnRemoteMixedAudioReceived(self, frame: Any) -> None:
            if not frame:
                return
            try:
                data = frame.PcmData()
                sr = getattr(frame, "sample_rate", None) or SAMPLE_RATE
                ch = getattr(frame, "channels", None) or CHANNELS
                if data:
                    self._edge._emit_audio_received("mixed", data, sr, ch)
            except Exception as e:
                logger.debug("OnRemoteMixedAudioReceived: %s", e)

    _TencentDelegate = _TencentDelegateCls
