"""Tencent TRTC media track implementations."""

import asyncio
import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Optional

import av
from getstream.video.rtc.track_util import PcmData
from vision_agents.plugins.tencent.video_utils import av_frame_to_yuv420p

if TYPE_CHECKING:
    import aiortc

logger = logging.getLogger(__name__)

try:
    from liteav import (
        AUDIO_CODEC_TYPE_PCM,
        STREAM_TYPE_VIDEO_HIGH,
        VIDEO_PIXEL_FORMAT_YUV420p,
        VIDEO_ROTATION_0,
        AudioFrame,
        PixelFrame,
    )
except ImportError:
    pass

FRAME_MS = 20
SAMPLE_RATE = 16000
CHANNELS = 1
BYTES_PER_20MS = 640

_DEFAULT_VIDEO_FPS = 15


class TencentAudioTrack:
    """Duck-typed audio track that sends PCM via Tencent SendAudioFrame.

    Frames are paced at 20 ms intervals using wall-clock time to match the
    real-time playback rate expected by the TRTC SDK.
    """

    _FRAME_INTERVAL_S = FRAME_MS / 1000.0
    _TAIL_FLUSH_S = _FRAME_INTERVAL_S * 2

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._cloud: Any = None
        self._running = True
        self._sender_thread: Optional[threading.Thread] = None
        self._pts = 10
        self._last_write_at = 0.0

    def set_cloud(self, cloud: Any) -> None:
        self._cloud = cloud
        if self._sender_thread is None:
            self._sender_thread = threading.Thread(target=self._send_loop, daemon=True)
            self._sender_thread.start()

    async def write(self, pcm: PcmData) -> None:
        if not self._running or pcm is None:
            return
        if pcm.sample_rate != SAMPLE_RATE or pcm.channels != CHANNELS:
            pcm = pcm.resample(
                target_sample_rate=SAMPLE_RATE, target_channels=CHANNELS
            )
        if pcm.samples is not None and pcm.samples.size > 0:
            data = pcm.samples.tobytes()
        else:
            data = pcm.to_bytes()
        if data:
            with self._lock:
                self._buffer.extend(data)
                self._last_write_at = time.monotonic()

    def stop(self) -> None:
        self._running = False

    async def flush(self) -> None:
        with self._lock:
            self._buffer.clear()

    def _pop_frame(self) -> bytes | None:
        with self._lock:
            buf_len = len(self._buffer)
            if buf_len >= BYTES_PER_20MS:
                frame_data = bytes(self._buffer[:BYTES_PER_20MS])
                del self._buffer[:BYTES_PER_20MS]
                return frame_data
            if buf_len > 0 and (time.monotonic() - self._last_write_at) > self._TAIL_FLUSH_S:
                frame_data = bytes(self._buffer) + b"\x00" * (BYTES_PER_20MS - buf_len)
                self._buffer.clear()
                return frame_data
            return None

    def _send_loop(self) -> None:
        next_frame_at = time.monotonic()

        while self._running and self._cloud is not None:
            now = time.monotonic()

            if now < next_frame_at:
                time.sleep(min(next_frame_at - now, 0.005))
                continue

            frame_bytes = self._pop_frame()
            if frame_bytes is None:
                time.sleep(0.005)
                continue

            frame = AudioFrame()
            frame.sample_rate = SAMPLE_RATE
            frame.channels = CHANNELS
            frame.bits_per_sample = 16
            frame.codec = AUDIO_CODEC_TYPE_PCM
            frame.pts = self._pts
            self._pts += FRAME_MS
            frame.SetData(frame_bytes)
            self._cloud.SendAudioFrame(frame)

            next_frame_at += self._FRAME_INTERVAL_S
            if next_frame_at < now:
                next_frame_at = now + self._FRAME_INTERVAL_S


class TencentIncomingVideoTrack:
    """Duck-typed VideoStreamTrack fed by TRTC OnRemotePixelFrameReceived.

    The core VideoForwarder reads frames via ``await track.recv()``.
    Frames are pushed from the TRTC delegate thread through the event loop.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._queue: asyncio.Queue[av.VideoFrame] = asyncio.Queue(maxsize=4)
        self._last_frame: av.VideoFrame | None = None
        self._state = "live"

    @property
    def readyState(self) -> str:  # noqa: N802
        return self._state

    async def recv(self) -> av.VideoFrame:
        try:
            frame = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            self._last_frame = frame
            return frame
        except asyncio.TimeoutError:
            if self._last_frame is not None:
                return self._last_frame
            return av.VideoFrame(width=640, height=480, format="yuv420p")

    def push_frame(self, frame: av.VideoFrame) -> None:
        """Thread-safe push from the TRTC callback thread."""
        try:
            self._queue.put_nowait(frame)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._queue.put_nowait(frame)
            except asyncio.QueueFull:
                pass

    def stop(self) -> None:
        self._state = "ended"


class TencentOutgoingVideoTrack:
    """Reads av.VideoFrames from an aiortc track and sends them via TRTC SendPixelFrame.

    The send loop runs as an asyncio task that calls ``recv()`` on the source
    track and converts each frame to YUV420p before handing it to the C SDK.
    """

    def __init__(self, source: "aiortc.MediaStreamTrack", fps: int = _DEFAULT_VIDEO_FPS):
        self._source = source
        self._fps = fps
        self._frame_interval = 1.0 / fps
        self._cloud: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[None] | None = None
        self._running = True
        self._pts = 10
        self._pts_start_time: float | None = None

    def set_cloud(self, cloud: Any, loop: asyncio.AbstractEventLoop) -> None:
        self._cloud = cloud
        self._loop = loop
        self._running = True
        self._pts = 10
        self._pts_start_time = time.monotonic()
        if self._task is None:
            self._task = loop.create_task(self._send_loop())

    async def _send_loop(self) -> None:
        next_send = asyncio.get_event_loop().time()
        if self._pts_start_time is None:
            self._pts_start_time = time.monotonic()
        while self._running and self._cloud is not None:
            try:
                frame = await self._source.recv()
            except Exception:
                logger.exception("Outgoing video recv failed")
                break

            yuv_bytes, width, height = av_frame_to_yuv420p(frame)

            pf = PixelFrame()
            pf.width = width
            pf.height = height
            pf.format = VIDEO_PIXEL_FORMAT_YUV420p
            pf.rotation = VIDEO_ROTATION_0
            elapsed_ms = int((time.monotonic() - self._pts_start_time) * 1000)
            wall_clock_pts = 10 + elapsed_ms
            pts = max(self._pts + 1, wall_clock_pts)
            pf.pts = pts
            self._pts = pts
            pf.SetData(yuv_bytes)
            self._cloud.SendPixelFrame(STREAM_TYPE_VIDEO_HIGH, pf)

            now = asyncio.get_event_loop().time()
            next_send += self._frame_interval
            if next_send < now:
                next_send = now + self._frame_interval
            delay = next_send - now
            if delay > 0:
                await asyncio.sleep(delay)

    def stop(self) -> None:
        self._running = False
        self._pts_start_time = None
        if self._task is not None:
            self._task.cancel()
            self._task = None
