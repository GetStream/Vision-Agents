"""Tencent TRTC media track implementations."""

import asyncio
import concurrent.futures
import logging
import queue
import threading
import time
from collections import deque
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Optional

import av
from getstream.video.rtc.track_util import PcmData
from vision_agents.plugins.tencent.video_utils import av_frame_to_yuv420p, yuv420p_to_av_frame

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


_SILENCE_FRAME = b"\x00" * BYTES_PER_20MS


class TencentAudioTrack:
    """Duck-typed audio track that sends PCM via Tencent SendAudioFrame.

    Frames are paced at 20 ms intervals using wall-clock time to match the
    real-time playback rate expected by the TRTC SDK.

    Incoming PCM is pre-chunked into 20 ms frames (640 bytes) in a deque so
    both enqueue and dequeue are O(1).

    The sender thread runs continuously at 20 ms cadence, sending silence when
    no real audio is queued.  This keeps the remote receiver's jitter buffer
    synchronised so the start of each speech segment is never clipped.

    The queue is intentionally unbounded: TTS engines produce audio in bursts
    (faster than real-time) and the sender thread drains at exactly real-time.
    """

    _FRAME_INTERVAL_S = FRAME_MS / 1000.0
    _TAIL_FLUSH_S = _FRAME_INTERVAL_S * 2

    _write_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="trtc-audio-write"
    )

    def __init__(self) -> None:
        self._queue: deque[bytes] = deque()
        self._remainder = bytearray()
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
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._write_executor, self._write_sync, pcm)

    def _write_sync(self, pcm: PcmData) -> None:
        if pcm.sample_rate != SAMPLE_RATE or pcm.channels != CHANNELS:
            pcm = pcm.resample(target_sample_rate=SAMPLE_RATE, target_channels=CHANNELS)
        if pcm.samples is not None and pcm.samples.size > 0:
            data = pcm.samples.tobytes()
        else:
            data = pcm.to_bytes()
        if not data:
            return
        with self._lock:
            if self._remainder:
                data = bytes(self._remainder) + data
                self._remainder.clear()
            offset = 0
            while offset + BYTES_PER_20MS <= len(data):
                self._queue.append(data[offset:offset + BYTES_PER_20MS])
                offset += BYTES_PER_20MS
            if offset < len(data):
                self._remainder.extend(data[offset:])
            self._last_write_at = time.monotonic()

    def stop(self) -> None:
        self._running = False

    async def flush(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._flush_sync)

    def _flush_sync(self) -> None:
        with self._lock:
            self._queue.clear()
            self._remainder.clear()

    def _pop_one(self) -> bytes:
        with self._lock:
            if self._queue:
                return self._queue.popleft()
            if (
                self._remainder
                and (time.monotonic() - self._last_write_at) > self._TAIL_FLUSH_S
            ):
                padded = (
                    bytes(self._remainder)
                    + b"\x00" * (BYTES_PER_20MS - len(self._remainder))
                )
                self._remainder.clear()
                return padded
            return _SILENCE_FRAME

    def _send_one(self, frame_bytes: bytes) -> bool:
        frame = AudioFrame()
        frame.sample_rate = SAMPLE_RATE
        frame.channels = CHANNELS
        frame.bits_per_sample = 16
        frame.codec = AUDIO_CODEC_TYPE_PCM
        frame.pts = self._pts
        self._pts += FRAME_MS
        frame.SetData(frame_bytes)
        try:
            self._cloud.SendAudioFrame(frame)
            return True
        except Exception:
            logger.exception("Tencent SendAudioFrame failed")
            return False

    def _send_loop(self) -> None:
        next_frame_at = time.monotonic()

        while self._running and self._cloud is not None:
            now = time.monotonic()

            if now < next_frame_at:
                time.sleep(next_frame_at - now)

            if not self._send_one(self._pop_one()):
                return

            next_frame_at += self._FRAME_INTERVAL_S
            # If we fell behind (GIL contention, etc.), skip to now instead
            # of bursting catch-up frames — the SDK drops bursts as
            # "same timestamp" packets.
            if next_frame_at < time.monotonic():
                next_frame_at = time.monotonic() + self._FRAME_INTERVAL_S


class TencentIncomingVideoTrack:
    """Duck-typed VideoStreamTrack fed by TRTC OnRemotePixelFrameReceived.

    The core VideoForwarder reads frames via ``await track.recv()``.
    Raw YUV bytes are pushed from the TRTC delegate thread into a stdlib
    thread-safe queue.  Conversion to av.VideoFrame happens lazily in recv()
    to avoid holding the GIL on the callback thread.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._queue: queue.Queue[tuple[bytes, int, int, int]] = queue.Queue(maxsize=4)
        self._last_frame: av.VideoFrame | None = None
        self._state = "live"

    @property
    def readyState(self) -> str:  # noqa: N802
        return self._state

    async def recv(self) -> av.VideoFrame:
        loop = asyncio.get_running_loop()
        try:
            frame = await loop.run_in_executor(None, self._recv_sync)
            self._last_frame = frame
            return frame
        except queue.Empty:
            if self._last_frame is not None:
                return self._last_frame
            return av.VideoFrame(width=640, height=480, format="yuv420p")

    def _recv_sync(self) -> av.VideoFrame:
        yuv_bytes, width, height, pts = self._queue.get(True, 1.0)
        frame = yuv420p_to_av_frame(yuv_bytes, width, height)
        frame.pts = pts
        frame.time_base = Fraction(1, 1000)
        return frame

    def push_frame(self, yuv_bytes: bytes, width: int, height: int, pts: int) -> None:
        """Thread-safe push from the TRTC callback thread."""
        try:
            self._queue.put_nowait((yuv_bytes, width, height, pts))
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait((yuv_bytes, width, height, pts))
            except queue.Full:
                pass

    def stop(self) -> None:
        self._state = "ended"


class TencentOutgoingVideoTrack:
    """Reads av.VideoFrames from an aiortc track and sends them via TRTC SendPixelFrame.

    The send loop runs as an asyncio task that calls ``recv()`` on the source
    track, then offloads YUV420p conversion and SDK send to a background thread
    to avoid blocking the event loop.
    """

    _video_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="trtc-video-send"
    )

    def __init__(
        self, source: "aiortc.MediaStreamTrack", fps: int = _DEFAULT_VIDEO_FPS
    ):
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

    def _send_frame(self, frame: "av.VideoFrame") -> None:
        yuv_bytes, width, height = av_frame_to_yuv420p(frame)

        pf = PixelFrame()
        pf.width = width
        pf.height = height
        pf.format = VIDEO_PIXEL_FORMAT_YUV420p
        pf.rotation = VIDEO_ROTATION_0
        assert self._pts_start_time is not None
        elapsed_ms = int((time.monotonic() - self._pts_start_time) * 1000)
        wall_clock_pts = 10 + elapsed_ms
        pts = max(self._pts + 1, wall_clock_pts)
        pf.pts = pts
        self._pts = pts
        pf.SetData(yuv_bytes)
        self._cloud.SendPixelFrame(STREAM_TYPE_VIDEO_HIGH, pf)

    async def _send_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._pts_start_time is None:
            self._pts_start_time = time.monotonic()
        next_send = loop.time()
        while self._running and self._cloud is not None:
            try:
                frame = await self._source.recv()
            except Exception:
                logger.exception("Outgoing video recv failed")
                break

            await loop.run_in_executor(self._video_executor, self._send_frame, frame)

            now = loop.time()
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
