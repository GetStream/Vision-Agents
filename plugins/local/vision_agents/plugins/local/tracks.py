"""
LocalTransport: audio/video track implementations.

Provides LocalOutputAudioTrack for speaker playback and LocalVideoTrack
for camera capture, enabling vision agents to run locally without cloud
edge infrastructure.
"""

import asyncio
import logging
import platform
import threading
import time
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


class LocalOutputAudioTrack(AudioStreamTrack):
    """Audio track that plays PcmData through an AudioOutputDevice.

    Uses an asyncio.Queue for backpressure: when the queue is full,
    ``write`` awaits until the playback task drains an item. The playback
    task offloads blocking device writes via ``asyncio.to_thread``.

    Extends AudioStreamTrack so it satisfies the MediaStreamTrack interface
    required by EdgeTransport.publish_tracks. Since this is a write-only
    (playback) track, recv() is not supported.
    """

    def __init__(self, audio_output: AudioOutputDevice, buffer_limit: int = 20):
        super().__init__()
        self._audio_output = audio_output
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=buffer_limit)
        self._running = False
        self._playback_task: asyncio.Task[None] | None = None
        self._write_lock = asyncio.Lock()

    async def recv(self) -> av.AudioFrame:
        """Not supported — this is a write-only playback track."""
        raise NotImplementedError(
            "LocalOutputAudioTrack is a playback-only track; recv() is not supported"
        )

    def start(self) -> None:
        """Start the audio output stream."""
        if self._running:
            return

        self._audio_output.start()
        self._running = True
        self._playback_task = asyncio.create_task(self._playback_loop())

    async def write(self, data: PcmData) -> None:
        """Write PCM data to be played on the speaker."""
        if not self._running:
            return

        async with self._write_lock:
            samples = self._process_audio(data)
            await self._queue.put(samples)

    async def flush(self) -> None:
        """Clear any pending audio data and abort OS-level playback."""
        async with self._write_lock:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            self._audio_output.flush()

    def stop(self) -> None:
        """Stop the audio output stream."""
        super().stop()
        self._running = False

        if self._playback_task is not None:
            self._playback_task.cancel()
            self._playback_task = None

        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self._audio_output.stop()

    async def _playback_loop(self) -> None:
        """Async task that drains the queue into the AudioOutput backend."""
        try:
            while True:
                data = await self._queue.get()
                try:
                    await asyncio.to_thread(self._audio_output.write, data)
                except sd.PortAudioError as err:
                    logger.debug("PortAudio playback error: %s", err)
        except asyncio.CancelledError:
            logger.debug("Playback loop cancelled")
            raise
        except ValueError:
            logger.exception("Audio data processing error")
        except OSError:
            logger.exception("Audio playback device error")

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

    def _read_frame(self, max_retries: int = 20, retry_timeout: float = 0.02) -> Any:
        """Read a single frame from the camera (blocking)."""
        if self._container is None:
            return None

        for attempt in range(max_retries):
            try:
                for packet in self._container.demux(self._stream):
                    for frame in packet.decode():
                        return frame
            except BlockingIOError:
                if attempt < max_retries - 1:
                    time.sleep(retry_timeout)
                    continue
                logger.debug("Camera not ready after %d retries", max_retries)
                return None
            except OSError:
                logger.warning("Error reading camera frame", exc_info=True)
                return None
        return None

    async def recv(self) -> av.VideoFrame:
        """Receive the next video frame."""
        if self._stopped:
            raise RuntimeError("Track has been stopped")

        if not self._started:
            self._started = True
            self._start_time = time.time()

            await asyncio.to_thread(self._open_camera)

        frame = await asyncio.to_thread(self._read_frame)

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
