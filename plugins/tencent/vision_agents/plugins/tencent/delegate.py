"""TRTCCloudDelegate subclass that forwards SDK callbacks to TencentEdge.

The class is defined only when liteav was importable; on non-linux
platforms ``TencentDelegate`` resolves to ``None`` and TencentEdge
fails earlier via ``require_liteav()``.

All callbacks are wrapped in try/except because any unhandled Python
exception in a SWIG director callback triggers
Swig::DirectorMethodException on the C++ side, which calls
std::terminate and kills the process.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from vision_agents.plugins.tencent.bindings import (
    STREAM_TYPE_VIDEO_HIGH,
    AudioEncodeParams,
    TRTCCloudDelegate,
    VideoEncodeParams,
)
from vision_agents.plugins.tencent.tracks import CHANNELS, SAMPLE_RATE

if TYPE_CHECKING:
    from vision_agents.plugins.tencent.tencent_edge import (
        TencentConnection,
        TencentEdge,
    )

logger = logging.getLogger(__name__)


def _extract_pcm(frame: Any) -> bytes:
    if frame.size <= 0:
        return b""
    return frame.data


TencentDelegate: Any = None
if TRTCCloudDelegate is not None:

    class _TencentDelegateCls(TRTCCloudDelegate):
        """Forwards TRTC SDK callbacks to TencentEdge and TencentConnection."""

        def __init__(
            self,
            edge: "TencentEdge",
            connection: "TencentConnection",
            loop: asyncio.AbstractEventLoop,
        ):
            # Intentionally not super().__init__(): TRTCCloudDelegate is a SWIG
            # director — its __init__ binds the C++ side via `self.this` and
            # registers Python callback dispatch. Tencent's bundled
            # python_x86_64 sample uses exactly this explicit form, and that
            # is the only invocation contract liteav documents. We follow it
            # verbatim rather than relying on cooperative MRO that the C++
            # binding never promised to honour.
            TRTCCloudDelegate.__init__(self)
            self._edge = edge
            self._connection = connection
            self._agent_user_id = edge._agent_user_id or ""

        def OnError(self, error: int) -> None:  # type: ignore[override]
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

        def OnConnectionStateChanged(self, old_state: int, new_state: int) -> None:  # type: ignore[override]
            logger.debug(
                "Tencent TRTC connection state: %s -> %s", old_state, new_state
            )

        def OnLocalAudioChannelDestroyed(self) -> None:
            pass

        def OnLocalVideoChannelCreated(self, stream_type: int) -> None:  # type: ignore[override]
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

        def OnLocalVideoChannelDestroyed(self, stream_type: int) -> None:  # type: ignore[override]
            pass

        def OnRequestChangeVideoEncodeBitrate(  # type: ignore[override]
            self, stream_type: int, bitrate_bps: int
        ) -> None:
            pass

        def OnRequestKeyFrame(self, stream_type: int) -> None:  # type: ignore[override]
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

        def OnRemoteVideoAvailable(  # type: ignore[override]
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

        def OnRemoteVideoFrameReceived(  # type: ignore[override]
            self, user_id: str, stream_type: int, frame: Any
        ) -> None:
            pass

        def OnRemotePixelFrameReceived(  # type: ignore[override]
            self, user_id: str, stream_type: int, frame: Any
        ) -> None:
            try:
                if not user_id or user_id == self._agent_user_id:
                    return
                width = frame.width
                height = frame.height
                if width <= 0 or height <= 0 or frame.size <= 0:
                    return
                self._edge._push_video_frame(
                    user_id, frame.data, width, height, frame.pts
                )
            except BaseException:
                logger.exception("OnRemotePixelFrameReceived callback failed")

        def OnSeiMessageReceived(  # type: ignore[override]
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

    TencentDelegate = _TencentDelegateCls
