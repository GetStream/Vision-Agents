"""
LocalTransport: audio/video track implementations.

Provides LocalOutputAudioTrack for speaker playback and LocalVideoTrack
for camera capture, enabling vision agents to run locally without cloud
edge infrastructure.
"""

import asyncio
import logging
import platform
import queue
import threading
import time
from collections import deque
from fractions import Fraction
from typing import Any

import av
import numpy as np
import sounddevice as sd
from aiortc import AudioStreamTrack, VideoStreamTrack
from getstream.video.rtc.track_util import PcmData

from .devices import AudioOutputDevice

logger = logging.getLogger(__name__)


def _get_camera_input_format() -> str:
    """Get the FFmpeg input format for the current platform."""
    system = platform.system()
    if system == "Darwin":
        return "avfoundation"
    elif system == "Linux":
        return "v4l2"
    elif system == "Windows":
        return "dshow"
    else:
        raise RuntimeError(f"Unsupported platform for camera capture: {system}")


class PlaybackBuffer:
    """Thread-safe audio buffer with duration-based limits.

    Drops oldest samples when the buffer exceeds the configured duration,
    keeping playback close to real-time. Uses threading primitives so the
    consumer can block on ``get`` from a plain thread.
    """

    def __init__(self, sample_rate: int, channels: int, buffer_limit_ms: int = 2000):
        self._sample_rate = sample_rate
        self._channels = channels
        self._buffer_limit_ms = buffer_limit_ms
        self._deque: deque[np.ndarray] = deque()
        self._total_samples = 0
        self._lock = threading.Lock()
        self._not_empty = threading.Event()

    def _max_samples(self) -> int:
        return int((self._buffer_limit_ms / 1000) * self._sample_rate * self._channels)

    def put(self, samples: np.ndarray) -> None:
        """Append samples, dropping oldest data if the buffer would exceed the limit."""
        with self._lock:
            self._deque.append(samples)
            self._total_samples += len(samples)

            limit = self._max_samples()
            dropped = 0
            while self._total_samples > limit and self._deque:
                oldest = self._deque.popleft()
                self._total_samples -= len(oldest)
                dropped += len(oldest)

            if dropped:
                dropped_ms = (dropped / self._channels / self._sample_rate) * 1000
                logger.warning(
                    "Playback buffer over %.0fms limit, dropped %.1fms of audio",
                    self._buffer_limit_ms,
                    dropped_ms,
                )

            self._not_empty.set()

    def get(self, timeout: float = 0.1) -> np.ndarray:
        """Block until data is available or *timeout* expires.

        Raises:
            queue.Empty: If no data arrives within the timeout.
        """
        if not self._not_empty.wait(timeout=timeout):
            raise queue.Empty()

        with self._lock:
            if not self._deque:
                self._not_empty.clear()
                raise queue.Empty()

            chunk = self._deque.popleft()
            self._total_samples -= len(chunk)

            if not self._deque:
                self._not_empty.clear()

            return chunk

    def flush(self) -> None:
        """Discard all buffered audio."""
        with self._lock:
            self._deque.clear()
            self._total_samples = 0
            self._not_empty.clear()

    def empty(self) -> bool:
        return self._total_samples == 0


class LocalOutputAudioTrack(AudioStreamTrack):
    """Audio track that plays PcmData through an AudioOutputDevice.

    Handles PcmData-to-numpy conversion, resampling, and queued playback
    on a dedicated thread.

    Extends AudioStreamTrack so it satisfies the MediaStreamTrack interface
    required by EdgeTransport.publish_tracks. Since this is a write-only
    (playback) track, recv() is not supported.
    """

    def __init__(self, audio_output: AudioOutputDevice, buffer_limit_ms: int = 30_000):
        super().__init__()
        self._audio_output = audio_output
        self._buffer = PlaybackBuffer(
            sample_rate=audio_output.sample_rate,
            channels=audio_output.channels,
            buffer_limit_ms=buffer_limit_ms,
        )
        self._running = False
        self._stopped = False
        self._playback_thread: threading.Thread | None = None
        self._write_lock = asyncio.Lock()

    async def recv(self) -> av.AudioFrame:
        """Not supported — this is a write-only playback track."""
        raise NotImplementedError(
            "LocalOutputAudioTrack is a playback-only track; recv() is not supported"
        )

    def start(self) -> None:
        """Start the audio output stream."""
        if self._running or self._stopped:
            return

        self._audio_output.start()
        self._running = True
        self._playback_thread = threading.Thread(
            target=self._playback_loop, daemon=True
        )
        self._playback_thread.start()

    async def write(self, data: PcmData) -> None:
        """Write PCM data to be played on the speaker."""
        if self._stopped:
            return

        async with self._write_lock:
            samples = self._process_audio(data)
            self._buffer.put(samples)

    async def flush(self) -> None:
        """Clear any pending audio data and abort OS-level playback."""
        async with self._write_lock:
            self._buffer.flush()
            self._audio_output.flush()

    def stop(self) -> None:
        """Stop the audio output stream."""
        super().stop()
        self._stopped = True
        self._running = False

        if self._playback_thread is not None:
            self._playback_thread.join(timeout=1.0)
            self._playback_thread = None

        self._audio_output.stop()

    def _playback_loop(self) -> None:
        """Dedicated thread that drains the buffer into the AudioOutput backend."""
        try:
            while self._running:
                try:
                    data = self._buffer.get(timeout=0.1)
                    self._audio_output.write(data)
                except queue.Empty:
                    continue
                except sd.PortAudioError as err:
                    logger.debug("PortAudio playback error: %s", err)
                    continue
        except ValueError:
            logger.exception("Audio data processing error")
        except OSError:
            logger.exception("Audio playback device error")
        finally:
            logger.info("Stopped audio output")

    def _process_audio(self, data: PcmData) -> np.ndarray:
        """Resample and convert PcmData to flat int16 numpy for the backend."""
        target_rate = self._audio_output.sample_rate
        target_channels = self._audio_output.channels

        if data.sample_rate != target_rate or data.channels != target_channels:
            data = data.resample(target_rate, target_channels)

        samples = data.to_int16().samples

        if samples.ndim == 2:
            samples = samples.T.flatten()

        return samples


class LocalVideoTrack(VideoStreamTrack):
    """Video track that captures from local camera using PyAV."""

    kind = "video"

    def __init__(
        self,
        device: str,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        super().__init__()

        self._device = device
        self._width = width
        self._height = height
        self._fps = fps
        self._container: Any = None
        self._stream: Any = None
        self._started = False
        self._stopped = False
        self._frame_count = 0
        self._start_time: float | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    def _open_camera(self) -> None:
        """Open the camera device with PyAV."""
        input_format = _get_camera_input_format()
        system = platform.system()

        options: dict[str, str] = {
            "framerate": str(self._fps),
        }

        if system == "Darwin":
            device_path = self._device
            options["video_size"] = f"{self._width}x{self._height}"
            options["pixel_format"] = "uyvy422"
        elif system == "Linux":
            device_path = self._device
            options["video_size"] = f"{self._width}x{self._height}"
        elif system == "Windows":
            device_path = self._device
            options["video_size"] = f"{self._width}x{self._height}"
        else:
            raise RuntimeError(f"Unsupported platform: {system}")

        self._container = av.open(
            device_path,
            format=input_format,
            options=options,
        )
        self._stream = self._container.streams.video[0]
        logger.info(
            "Opened camera: %s (%dx%d @ %dfps)",
            self._device,
            self._width,
            self._height,
            self._fps,
        )

    def _read_frame(self) -> Any:
        """Read a single frame from the camera (blocking)."""
        if self._container is None:
            return None

        try:
            for packet in self._container.demux(self._stream):
                for frame in packet.decode():
                    return frame
        except OSError:
            logger.warning("Error reading camera frame")
            return None
        return None

    async def recv(self) -> Any:
        """Receive the next video frame."""
        if self._stopped:
            raise RuntimeError("Track has been stopped")

        if not self._started:
            self._started = True
            self._start_time = time.time()
            self._loop = asyncio.get_running_loop()
            await self._loop.run_in_executor(None, self._open_camera)

        assert self._loop is not None
        frame = await self._loop.run_in_executor(None, self._read_frame)

        if frame is None:
            frame = av.VideoFrame(
                width=self._width, height=self._height, format="rgb24"
            )
            frame.planes[0].update(bytes(self._width * self._height * 3))

        self._frame_count += 1
        frame.pts = self._frame_count
        frame.time_base = Fraction(1, self._fps)
        return frame

    def stop(self) -> None:
        """Stop camera capture and release resources."""
        with self._lock:
            self._stopped = True
            if self._container is not None:
                try:
                    self._container.close()
                except OSError:
                    logger.warning("Error closing camera")
                self._container = None
                self._stream = None
            logger.info("Stopped camera capture")
