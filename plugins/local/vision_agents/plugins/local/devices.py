"""
Device enumeration and selection utilities for LocalTransport.

Provides typed device representations and interactive prompts for
selecting audio and video devices when running agents locally.
"""

import glob
import logging
import platform
import queue
import re
import subprocess
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from .utils import prompt_selection

logger = logging.getLogger(__name__)


_AVFOUNDATION_RE = re.compile(r"\[AVFoundation.*?\]\s*\[(\d+)\]\s*(.+)")
_DSHOW_DEVICE_RE = re.compile(r'"(.+?)"')


class AudioInputDevice:
    """Audio input device (microphone).

    Combines device metadata with stream capture. Subclass to implement
    custom audio backends (e.g. GStreamer).
    """

    def __init__(
        self,
        index: int,
        name: str,
        sample_rate: int = 48000,
        channels: int = 1,
        is_default: bool = False,
        blocksize: int | None = None,
    ):
        self.index = index
        self.name = name
        self._sample_rate = sample_rate
        self._channels = channels
        self.is_default = is_default
        self._blocksize = (
            blocksize if blocksize is not None else int(sample_rate * 0.02)
        )
        self._stream: sd.InputStream | None = None
        self._buffer: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        if status:
            logger.warning("Audio input status: %s", status)
        try:
            self._buffer.put_nowait(indata.copy())
        except queue.Full:
            pass

    def start(self) -> None:
        """Open and start the audio input stream."""
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="int16",
            blocksize=self._blocksize,
            device=self.index,
            callback=self._callback,
        )
        self._stream.start()
        logger.info(
            "Started audio input: %dHz, %d channels",
            self._sample_rate,
            self._channels,
        )

    def read(self) -> np.ndarray | None:
        """Block until audio data is available (up to 100ms timeout)."""
        try:
            return self._buffer.get(timeout=0.1)
        except queue.Empty:
            return None

    def stop(self) -> None:
        """Stop and close the audio input stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Stopped audio input")


class AudioOutputDevice:
    """Audio output device (speaker/headphones).

    Combines device metadata with stream playback. Subclass to implement
    custom audio backends (e.g. GStreamer).
    """

    def __init__(
        self,
        index: int,
        name: str,
        sample_rate: int = 48000,
        channels: int = 2,
        is_default: bool = False,
        blocksize: int = 2048,
    ):
        self.index = index
        self.name = name
        self._sample_rate = sample_rate
        self._channels = channels
        self.is_default = is_default
        self._blocksize = blocksize
        self._stream: sd.OutputStream | None = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    def start(self) -> None:
        """Open and start the audio output stream."""
        self._stream = sd.OutputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="int16",
            blocksize=self._blocksize,
            device=self.index,
        )
        self._stream.start()
        logger.info(
            "Started audio output: %dHz, %d channels",
            self._sample_rate,
            self._channels,
        )

    def write(self, samples: np.ndarray) -> None:
        """Write flat int16 samples to the device."""
        if self._stream is None:
            return
        frames = len(samples) // self._channels
        audio = samples.reshape(frames, self._channels)
        self._stream.write(audio)

    def flush(self) -> None:
        """Abort current playback and restart the stream."""
        if self._stream is not None:
            self._stream.abort()
            self._stream.start()

    def stop(self) -> None:
        """Stop and close the audio output stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Stopped audio output")


@dataclass(frozen=True)
class CameraDevice:
    """A detected camera."""

    index: int
    name: str
    device: str


def list_audio_input_devices() -> list[AudioInputDevice]:
    """Return all audio input devices."""
    raw = sd.query_devices()
    default_in = sd.default.device[0]
    return [
        AudioInputDevice(
            index=i,
            name=dev["name"],
            sample_rate=int(dev["default_samplerate"]),
            channels=dev["max_input_channels"],
            is_default=(i == default_in),
        )
        for i, dev in enumerate(raw)
        if dev["max_input_channels"] > 0
    ]


def list_audio_output_devices() -> list[AudioOutputDevice]:
    """Return all audio output devices."""
    raw = sd.query_devices()
    default_out = sd.default.device[1]
    return [
        AudioOutputDevice(
            index=i,
            name=dev["name"],
            sample_rate=int(dev["default_samplerate"]),
            channels=dev["max_output_channels"],
            is_default=(i == default_out),
        )
        for i, dev in enumerate(raw)
        if dev["max_output_channels"] > 0
    ]


def select_audio_input_device() -> AudioInputDevice | None:
    """Interactive prompt to select an audio input device."""
    devices = list_audio_input_devices()
    default = next((d for d in devices if d.is_default), None)
    return prompt_selection(
        items=devices,
        formatter=_format_audio_device,
        header="INPUT DEVICES (Microphones)",
        default=default,
    )


def select_audio_output_device() -> AudioOutputDevice | None:
    """Interactive prompt to select an audio output device."""
    devices = list_audio_output_devices()
    default = next((d for d in devices if d.is_default), None)
    return prompt_selection(
        items=devices,
        formatter=_format_audio_device,
        header="OUTPUT DEVICES (Speakers)",
        default=default,
    )


def select_video_device() -> CameraDevice | None:
    """Interactive prompt to select a camera or skip.

    Returns:
        The selected camera device, or None if skipped.
    """
    cameras = list_cameras()

    return prompt_selection(
        items=cameras,
        formatter=lambda c: c.name,
        header="VIDEO DEVICES (Cameras)",
        allow_skip=True,
        empty_message="No cameras detected\n  (Camera support requires ffmpeg to be installed)",
    )


def list_cameras() -> list[CameraDevice]:
    """List available cameras on the system."""
    system = platform.system()

    if system == "Darwin":
        return _list_cameras_darwin()
    if system == "Linux":
        return _list_cameras_linux()
    if system == "Windows":
        return _list_cameras_windows()

    return []


def _format_audio_device(dev: AudioInputDevice | AudioOutputDevice) -> str:
    """Format an audio device for display."""
    default = " [DEFAULT]" if dev.is_default else ""
    return f"{dev.name} ({dev.sample_rate}Hz){default}"


def _list_cameras_darwin() -> list[CameraDevice]:
    """List cameras on macOS via ffmpeg/AVFoundation."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.warning("Failed to list cameras (is ffmpeg installed?)")
        return []

    cameras: list[CameraDevice] = []
    in_video_section = False

    for line in result.stderr.splitlines():
        if "AVFoundation video devices:" in line:
            in_video_section = True
            continue
        if "AVFoundation audio devices:" in line:
            break
        if in_video_section:
            match = _AVFOUNDATION_RE.search(line)
            if match:
                cam_idx = int(match.group(1))
                cameras.append(
                    CameraDevice(
                        index=cam_idx, name=match.group(2), device=str(cam_idx)
                    )
                )

    return cameras


def _list_cameras_linux() -> list[CameraDevice]:
    """List cameras on Linux via /dev/video* and sysfs."""
    cameras: list[CameraDevice] = []

    for i, dev_path in enumerate(sorted(glob.glob("/dev/video*"))):
        name_path = f"/sys/class/video4linux/{dev_path.split('/')[-1]}/name"
        try:
            with open(name_path) as f:
                name = f.read().strip()
        except OSError:
            name = dev_path
        cameras.append(CameraDevice(index=i, name=name, device=dev_path))

    return cameras


def _list_cameras_windows() -> list[CameraDevice]:
    """List cameras on Windows via ffmpeg/DirectShow."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-f", "dshow", "-list_devices", "true", "-i", "dummy"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.warning("Failed to list cameras (is ffmpeg installed?)")
        return []

    cameras: list[CameraDevice] = []
    in_video_section = False

    for line in result.stderr.splitlines():
        if "DirectShow video devices" in line:
            in_video_section = True
            continue
        if "DirectShow audio devices" in line:
            break
        if in_video_section:
            match = _DSHOW_DEVICE_RE.search(line)
            if match:
                name = match.group(1)
                cameras.append(
                    CameraDevice(
                        index=len(cameras),
                        name=name,
                        device=f'video="{name}"',
                    )
                )

    return cameras
