"""
LocalTransport: EdgeTransport implementation for local audio/video I/O.

Uses sounddevice for microphone input and speaker output, and PyAV for
camera capture, enabling vision agents to run locally without cloud
edge infrastructure.
"""

from __future__ import annotations

import asyncio
import logging
import platform
import queue
import subprocess
import threading
import time
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

try:
    import sounddevice as sd

    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    sd = None  # type: ignore[assignment]
    SOUNDDEVICE_AVAILABLE = False

try:
    import av

    PYAV_AVAILABLE = True
except ImportError:
    av = None  # type: ignore[assignment]
    PYAV_AVAILABLE = False

try:
    from aiortc import VideoStreamTrack

    AIORTC_AVAILABLE = True
except ImportError:
    VideoStreamTrack = object  # type: ignore[assignment, misc]
    AIORTC_AVAILABLE = False

from getstream.video.rtc.track_util import AudioFormat, PcmData
from pyee.asyncio import AsyncIOEventEmitter

from vision_agents.core.edge.edge_transport import EdgeTransport
from vision_agents.core.edge.events import AudioReceivedEvent
from vision_agents.core.edge.types import Connection, OutputAudioTrack, User
from vision_agents.core.events.manager import EventManager

if TYPE_CHECKING:
    from vision_agents.core.agents.agents import Agent

logger = logging.getLogger(__name__)


def _check_sounddevice() -> None:
    """Raise ImportError if sounddevice is not available."""
    if not SOUNDDEVICE_AVAILABLE:
        raise ImportError(
            "sounddevice is required for LocalTransport. "
            "Install it with: pip install sounddevice"
        )


def _check_pyav() -> None:
    """Raise ImportError if PyAV is not available."""
    if not PYAV_AVAILABLE:
        raise ImportError(
            "PyAV is required for camera support. "
            "Install it with: pip install av"
        )


def _check_aiortc() -> None:
    """Raise ImportError if aiortc is not available."""
    if not AIORTC_AVAILABLE:
        raise ImportError(
            "aiortc is required for video track support. "
            "Install it with: pip install aiortc"
        )


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


class LocalParticipant:
    """Represents the local user for audio events."""

    def __init__(self, user_id: str = "local-user"):
        self.user_id = user_id
        self.session_id = "local-session"


class LocalOutputAudioTrack:
    """
    Audio track that plays PCM data to the speaker using sounddevice.

    Uses a dedicated playback thread with blocking writes for reliable audio output.
    Implements the OutputAudioTrack protocol for compatibility with the Agent.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        channels: int = 2,
        blocksize: int = 2048,
        device: Optional[int] = None,
    ):
        _check_sounddevice()

        self._sample_rate = sample_rate
        self._channels = channels
        self._blocksize = blocksize
        self._device = device
        # Use a queue with reasonable max size to prevent memory buildup
        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=100)
        self._stream: Any = None  # sd.OutputStream when initialized
        self._running = False
        self._stopped = False
        self._playback_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _playback_loop(self) -> None:
        """Dedicated thread for audio playback using blocking writes."""
        try:
            # Open stream in blocking mode (no callback)
            with sd.OutputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype="int16",
                blocksize=self._blocksize,
                device=self._device,
            ) as stream:
                logger.info(
                    f"Started audio output: {self._sample_rate}Hz, "
                    f"{self._channels} channels"
                )

                while self._running:
                    try:
                        # Wait for audio data with timeout
                        data = self._queue.get(timeout=0.1)
                        if data is None:
                            # Sentinel value to stop
                            break

                        # Reshape to (frames, channels) for sounddevice
                        frames = len(data) // self._channels
                        audio = data.reshape(frames, self._channels)

                        # Blocking write - sounddevice handles timing
                        stream.write(audio)

                    except queue.Empty:
                        continue

        except Exception as e:
            logger.error(f"Audio playback error: {e}")
        finally:
            logger.info("Stopped audio output")

    def start(self) -> None:
        """Start the audio output stream."""
        if self._running or self._stopped:
            return

        self._running = True
        self._playback_thread = threading.Thread(
            target=self._playback_loop, daemon=True
        )
        self._playback_thread.start()

    def _process_audio(self, data: PcmData) -> np.ndarray:
        """Process audio data (resample and convert) - runs in thread pool."""
        # Resample if needed to match output sample rate and channels
        if data.sample_rate != self._sample_rate or data.channels != self._channels:
            data = data.resample(self._sample_rate, self._channels)

        # Ensure int16 format
        samples = data.to_int16().samples

        # PcmData.resample() returns shape (channels, samples) for stereo
        # We need to convert to interleaved format (samples, channels) then flatten
        if samples.ndim == 2:
            # Transpose from (channels, samples) to (samples, channels) and flatten
            samples = samples.T.flatten()

        return samples

    async def write(self, data: PcmData) -> None:
        """Write PCM data to be played on the speaker."""
        if self._stopped:
            return

        # Get or store the event loop
        if self._loop is None:
            self._loop = asyncio.get_running_loop()

        # Run CPU-intensive resampling in thread pool to avoid blocking event loop
        samples = await self._loop.run_in_executor(None, self._process_audio, data)

        # Non-blocking put with timeout to avoid blocking the event loop
        try:
            self._queue.put_nowait(samples)
        except queue.Full:
            logger.warning("Audio queue full, dropping samples")

    def stop(self) -> None:
        """Stop the audio output stream."""
        self._stopped = True
        self._running = False

        # Send sentinel to stop playback thread
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

        if self._playback_thread is not None:
            self._playback_thread.join(timeout=1.0)
            self._playback_thread = None

    async def flush(self) -> None:
        """Clear any pending audio data."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


class LocalVideoTrack(VideoStreamTrack):
    """
    Video track that captures from local camera using PyAV.

    Extends aiortc.VideoStreamTrack to provide camera frames for the agent
    video pipeline.
    """

    kind = "video"

    def __init__(
        self,
        device: str,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        """
        Initialize camera capture.

        Args:
            device: Device identifier (platform-specific)
            width: Capture width in pixels
            height: Capture height in pixels
            fps: Target frames per second
        """
        _check_pyav()
        _check_aiortc()
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
        self._start_time: Optional[float] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = threading.Lock()

    def _open_camera(self) -> None:
        """Open the camera device with PyAV."""
        input_format = _get_camera_input_format()
        system = platform.system()

        # Build input options
        options: dict[str, str] = {
            "framerate": str(self._fps),
        }

        # Platform-specific device path formatting
        if system == "Darwin":
            # macOS avfoundation: device index or name
            device_path = self._device
            options["video_size"] = f"{self._width}x{self._height}"
            options["pixel_format"] = "uyvy422"  # Common macOS format
        elif system == "Linux":
            # Linux v4l2: /dev/video* path
            device_path = self._device
            options["video_size"] = f"{self._width}x{self._height}"
        elif system == "Windows":
            # Windows dshow: video="Device Name"
            device_path = self._device
            options["video_size"] = f"{self._width}x{self._height}"
        else:
            raise RuntimeError(f"Unsupported platform: {system}")

        try:
            self._container = av.open(
                device_path,
                format=input_format,
                options=options,
            )
            self._stream = self._container.streams.video[0]
            logger.info(
                f"Opened camera: {self._device} ({self._width}x{self._height} @ {self._fps}fps)"
            )
        except Exception as e:
            logger.error(f"Failed to open camera {self._device}: {e}")
            raise

    def _read_frame(self) -> Optional[Any]:
        """Read a single frame from the camera (blocking)."""
        if self._container is None:
            return None

        try:
            for packet in self._container.demux(self._stream):
                for frame in packet.decode():
                    return frame
        except Exception as e:
            logger.warning(f"Error reading camera frame: {e}")
            return None
        return None

    async def recv(self) -> Any:
        """
        Receive the next video frame.

        Returns:
            av.VideoFrame with proper timestamp
        """
        if self._stopped:
            raise Exception("Track has been stopped")

        # Initialize on first call
        if not self._started:
            self._started = True
            self._start_time = time.time()
            if self._loop is None:
                self._loop = asyncio.get_running_loop()

            # Open camera in thread pool to avoid blocking
            await self._loop.run_in_executor(None, self._open_camera)

        # Read frame in thread pool to avoid blocking event loop
        frame = await self._loop.run_in_executor(None, self._read_frame)

        if frame is None:
            # Return a black frame if capture failed
            frame = av.VideoFrame(width=self._width, height=self._height, format="rgb24")
            frame.planes[0].update(bytes(self._width * self._height * 3))

        # Set proper timestamps for WebRTC
        self._frame_count += 1
        frame.pts = self._frame_count
        frame.time_base = av.Fraction(1, self._fps)

        return frame

    def stop(self) -> None:
        """Stop camera capture and release resources."""
        with self._lock:
            self._stopped = True
            if self._container is not None:
                try:
                    self._container.close()
                except Exception as e:
                    logger.warning(f"Error closing camera: {e}")
                self._container = None
                self._stream = None
            logger.info("Stopped camera capture")


class LocalConnection(Connection):
    """Connection wrapper for local transport."""

    def __init__(self, transport: "LocalTransport"):
        super().__init__()
        self._transport = transport
        self._participant_joined = asyncio.Event()
        # Local user is always "joined"
        self._participant_joined.set()

    def idle_since(self) -> float:
        """Local transport is never idle."""
        return 0.0

    async def wait_for_participant(self, timeout: Optional[float] = None) -> None:
        """Local user is always present, return immediately."""
        return

    async def close(self, timeout: float = 2.0) -> None:
        """Close the local connection."""
        await self._transport._stop_audio()


class LocalTransport(EdgeTransport):
    """
    EdgeTransport implementation for local audio/video I/O.

    Uses sounddevice for microphone input and speaker output, and PyAV for
    camera capture. This enables running vision agents locally without cloud
    dependencies.

    Example:
        transport = LocalTransport()
        agent = Agent(
            edge=transport,
            agent_user=User(name="Local AI", id="agent"),
            ...
        )
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        input_channels: int = 1,
        output_channels: int = 2,
        blocksize: int = 1024,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
        video_device: Optional[str] = None,
        video_width: int = 640,
        video_height: int = 480,
        video_fps: int = 30,
    ):
        """
        Initialize LocalTransport.

        Args:
            sample_rate: Audio sample rate in Hz (default: 48000)
            input_channels: Number of input channels (default: 1 for mono mic)
            output_channels: Number of output channels (default: 2 for stereo)
            blocksize: Audio block size in frames (default: 1024)
            input_device: Input device index (None for default)
            output_device: Output device index (None for default)
            video_device: Video device identifier (None to disable camera)
            video_width: Camera capture width (default: 640)
            video_height: Camera capture height (default: 480)
            video_fps: Camera capture FPS (default: 30)
        """
        super().__init__()
        _check_sounddevice()

        self._sample_rate = sample_rate
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._blocksize = blocksize
        self._input_device = input_device
        self._output_device = output_device

        # Video settings
        self._video_device = video_device
        self._video_width = video_width
        self._video_height = video_height
        self._video_fps = video_fps

        self.events = EventManager()
        # Register edge events for proper event emission
        from vision_agents.core.edge import events as edge_events

        self.events.register_events_from_module(edge_events)
        self._local_participant = LocalParticipant()

        self._input_stream: Any = None  # sd.InputStream when initialized
        self._input_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._mic_task: Optional[asyncio.Task[None]] = None
        self._running = False
        self._audio_track: Optional[LocalOutputAudioTrack] = None
        self._video_track: Optional[LocalVideoTrack] = None
        self._connection: Optional[LocalConnection] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def _microphone_callback_async(self, data: np.ndarray) -> None:
        """Process microphone data and emit AudioReceivedEvent."""
        # Convert to PcmData
        samples = data.flatten().astype(np.int16)
        pcm = PcmData(
            samples=samples,
            sample_rate=self._sample_rate,
            format=AudioFormat.S16,
            channels=self._input_channels,
        )
        pcm.participant = self._local_participant

        # Emit audio received event
        self.events.send(
            AudioReceivedEvent(
                plugin_name="local_transport",
                pcm_data=pcm,
                participant=self._local_participant,
            )
        )

    def _microphone_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Any,  # sd.CallbackFlags
    ) -> None:
        """Sounddevice callback for microphone input."""
        if status:
            logger.warning(f"Audio input status: {status}")

        if self._running and self._loop is not None:
            # Copy data to avoid issues with buffer reuse
            self._loop.call_soon_threadsafe(
                self._input_queue.put_nowait, indata.copy()
            )

    async def _microphone_loop(self) -> None:
        """Process microphone input from the queue."""
        try:
            while self._running:
                try:
                    data = await asyncio.wait_for(
                        self._input_queue.get(), timeout=0.1
                    )
                    await self._microphone_callback_async(data)
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            logger.debug("Microphone loop cancelled")
            raise

    async def _start_audio(self) -> None:
        """Start microphone capture."""
        if self._running:
            return

        self._running = True
        # Store the event loop reference for use in callbacks from other threads
        self._loop = asyncio.get_running_loop()

        # Start input stream
        self._input_stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._input_channels,
            dtype="int16",
            blocksize=self._blocksize,
            device=self._input_device,
            callback=self._microphone_callback,
        )
        self._input_stream.start()
        logger.info(
            f"Started microphone: {self._sample_rate}Hz, {self._input_channels} channels"
        )

        # Start processing loop
        self._mic_task = asyncio.create_task(self._microphone_loop())

    async def _stop_audio(self) -> None:
        """Stop all audio and video streams."""
        self._running = False

        # Stop microphone
        if self._mic_task is not None:
            self._mic_task.cancel()
            try:
                await self._mic_task
            except asyncio.CancelledError:
                pass
            self._mic_task = None

        if self._input_stream is not None:
            self._input_stream.stop()
            self._input_stream.close()
            self._input_stream = None
            logger.info("Stopped microphone")

        # Stop output track
        if self._audio_track is not None:
            self._audio_track.stop()

        # Stop video track
        if self._video_track is not None:
            self._video_track.stop()
            self._video_track = None

    async def publish_tracks(
        self,
        audio_track: Optional[OutputAudioTrack],
        video_track: Any,
    ) -> None:
        """
        Publish the agent's media tracks to participants.

        Audio Direction: OUTPUT - After this call, audio written to the
        audio_track will be sent to participants (played on their speakers).

        For LocalTransport, this starts the audio output stream and video capture.
        """
        if audio_track is not None and isinstance(audio_track, LocalOutputAudioTrack):
            audio_track.start()
            logger.info("Audio track published and started")

        if video_track is not None and isinstance(video_track, LocalVideoTrack):
            # Video track starts on first recv() call
            logger.info("Video track published")

    def create_output_audio_track(
        self, framerate: int = 48000, stereo: bool = True
    ) -> OutputAudioTrack:
        """
        Create an audio track for the agent's outgoing audio.

        Audio Direction: OUTPUT - The returned track is where the agent writes
        TTS/speech audio. Call publish_tracks() to start sending to participants.

        For LocalTransport, this creates a track that plays audio on the speaker.
        """
        channels = 2 if stereo else 1
        self._audio_track = LocalOutputAudioTrack(
            sample_rate=framerate,
            channels=channels,
            blocksize=self._blocksize,
            device=self._output_device,
        )
        return self._audio_track

    def create_video_track(self) -> Optional[LocalVideoTrack]:
        """
        Create a video track for the agent's camera input.

        Returns:
            LocalVideoTrack if a video device is configured, None otherwise.
        """
        if self._video_device is None:
            logger.debug("No video device configured, skipping video track creation")
            return None

        if not PYAV_AVAILABLE or not AIORTC_AVAILABLE:
            logger.warning(
                "PyAV or aiortc not available, skipping video track creation"
            )
            return None

        self._video_track = LocalVideoTrack(
            device=self._video_device,
            width=self._video_width,
            height=self._video_height,
            fps=self._video_fps,
        )
        return self._video_track

    def subscribe_to_track(self, track_id: str) -> None:
        """
        Subscribe to receive a remote participant's video track.

        Audio Direction: INPUT (video only) - Not implemented for local transport.
        """
        # No video support yet
        return None

    async def connect(self, agent: "Agent", call: Any = None) -> LocalConnection:
        """
        Connect to a call or room.

        Audio Direction: INPUT - After this call, audio from the microphone will
        be delivered via AudioReceivedEvent on the transport's event manager.

        For LocalTransport, this starts microphone capture.

        Args:
            agent: The agent joining
            call: Ignored for local transport (no actual call to join)

        Returns:
            LocalConnection for managing the connection
        """
        # Start microphone capture
        await self._start_audio()

        self._connection = LocalConnection(self)
        return self._connection

    async def disconnect(self) -> None:
        """
        Disconnect from the call and release all resources.

        Audio Direction: Stops both INPUT and OUTPUT streams.
        """
        await self._stop_audio()
        self._connection = None

    async def register_user(self, user: User) -> None:
        """
        Register the agent's user identity with the provider.

        Audio Direction: N/A - No-op for local transport.
        """
        pass

    def open_demo(self, *args: Any, **kwargs: Any) -> None:
        """
        Open a demo UI for testing (provider-specific).

        Audio Direction: N/A - Not supported for local transport.
        """
        logger.info(
            "LocalTransport does not have a demo UI. "
            "Audio is captured from your microphone and played on your speakers."
        )

    async def create_chat_channel(
        self, call: Any, user: User, instructions: str
    ) -> None:
        """
        Create a text chat channel associated with the call.

        Audio Direction: N/A - Not supported for local transport.
        """
        # No chat support for local transport
        return None


def list_audio_devices() -> None:
    """Print available audio devices for debugging."""
    if sd is None:
        print("sounddevice not installed")
        return

    print("Available audio devices:")
    print(sd.query_devices())
    print(f"\nDefault input device: {sd.default.device[0]}")
    print(f"Default output device: {sd.default.device[1]}")


def select_audio_devices() -> tuple[Optional[int], Optional[int]]:
    """
    Interactive prompt to select audio input and output devices.

    Returns:
        Tuple of (input_device_index, output_device_index).
        Returns None for either if using default.
    """
    _check_sounddevice()

    devices = sd.query_devices()
    default_in = sd.default.device[0]
    default_out = sd.default.device[1]

    # Collect input devices
    input_devices = []
    print("\n" + "=" * 50)
    print("INPUT DEVICES (Microphones)")
    print("=" * 50)
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            is_default = " [DEFAULT]" if i == default_in else ""
            print(f"  {len(input_devices)}: {dev['name']} ({int(dev['default_samplerate'])}Hz){is_default}")
            input_devices.append(i)

    # Collect output devices
    output_devices = []
    print("\n" + "=" * 50)
    print("OUTPUT DEVICES (Speakers)")
    print("=" * 50)
    for i, dev in enumerate(devices):
        if dev["max_output_channels"] > 0:
            is_default = " [DEFAULT]" if i == default_out else ""
            print(f"  {len(output_devices)}: {dev['name']} ({int(dev['default_samplerate'])}Hz){is_default}")
            output_devices.append(i)

    print("\n" + "-" * 50)

    # Select input device
    while True:
        try:
            choice = input(f"Select INPUT device [0-{len(input_devices)-1}] (Enter for default): ").strip()
            if choice == "":
                input_device = None
                print(f"  -> Using default: {devices[default_in]['name']}")
                break
            idx = int(choice)
            if 0 <= idx < len(input_devices):
                input_device = input_devices[idx]
                print(f"  -> Selected: {devices[input_device]['name']}")
                break
            print(f"  Invalid choice, enter 0-{len(input_devices)-1} or press Enter")
        except ValueError:
            print("  Please enter a number or press Enter")

    # Select output device
    while True:
        try:
            choice = input(f"Select OUTPUT device [0-{len(output_devices)-1}] (Enter for default): ").strip()
            if choice == "":
                output_device = None
                print(f"  -> Using default: {devices[default_out]['name']}")
                break
            idx = int(choice)
            if 0 <= idx < len(output_devices):
                output_device = output_devices[idx]
                print(f"  -> Selected: {devices[output_device]['name']}")
                break
            print(f"  Invalid choice, enter 0-{len(output_devices)-1} or press Enter")
        except ValueError:
            print("  Please enter a number or press Enter")

    print("-" * 50 + "\n")
    return input_device, output_device


def get_device_sample_rate(device_index: Optional[int], is_input: bool = True) -> int:
    """Get the default sample rate for a device."""
    _check_sounddevice()

    if device_index is None:
        device_index = sd.default.device[0 if is_input else 1]

    device_info = sd.query_devices(device_index)
    return int(device_info["default_samplerate"])


def list_cameras() -> list[dict[str, Any]]:
    """
    List available cameras on the system.

    Returns:
        List of camera info dicts with 'index', 'name', and 'device' keys.
    """
    _check_pyav()

    cameras: list[dict[str, Any]] = []
    system = platform.system()

    if system == "Darwin":
        # macOS: Use ffmpeg to list avfoundation devices
        try:
            result = subprocess.run(
                ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # Parse stderr for video devices
            output = result.stderr
            in_video_section = False
            for line in output.split("\n"):
                if "AVFoundation video devices:" in line:
                    in_video_section = True
                    continue
                if "AVFoundation audio devices:" in line:
                    break
                if in_video_section and "[AVFoundation" in line:
                    # Parse lines like: [AVFoundation @ 0x...] [0] FaceTime HD Camera
                    parts = line.split("]")
                    if len(parts) >= 3:
                        idx_part = parts[1].strip()
                        name_part = parts[2].strip()
                        if idx_part.startswith("[") and idx_part.endswith(""):
                            try:
                                idx = int(idx_part.strip("[]"))
                                cameras.append({
                                    "index": idx,
                                    "name": name_part,
                                    "device": str(idx),
                                })
                            except ValueError:
                                pass
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"Failed to list cameras: {e}")

    elif system == "Linux":
        # Linux: Check /dev/video* devices
        import glob

        video_devices = sorted(glob.glob("/dev/video*"))
        for i, dev_path in enumerate(video_devices):
            try:
                # Try to get device name from v4l2
                name_path = f"/sys/class/video4linux/{dev_path.split('/')[-1]}/name"
                try:
                    with open(name_path) as f:
                        name = f.read().strip()
                except FileNotFoundError:
                    name = dev_path
                cameras.append({
                    "index": i,
                    "name": name,
                    "device": dev_path,
                })
            except Exception:
                cameras.append({
                    "index": i,
                    "name": dev_path,
                    "device": dev_path,
                })

    elif system == "Windows":
        # Windows: Use ffmpeg to list dshow devices
        try:
            result = subprocess.run(
                ["ffmpeg", "-f", "dshow", "-list_devices", "true", "-i", "dummy"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = result.stderr
            in_video_section = False
            idx = 0
            for line in output.split("\n"):
                if "DirectShow video devices" in line:
                    in_video_section = True
                    continue
                if "DirectShow audio devices" in line:
                    break
                if in_video_section and '\"' in line:
                    # Parse lines like: [dshow @ 0x...] "Camera Name"
                    start = line.find('"')
                    end = line.rfind('"')
                    if start != -1 and end > start:
                        name = line[start + 1 : end]
                        cameras.append({
                            "index": idx,
                            "name": name,
                            "device": f'video="{name}"',
                        })
                        idx += 1
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"Failed to list cameras: {e}")

    return cameras


def select_video_device() -> Optional[str]:
    """
    Interactive prompt to select a camera or skip.

    Returns:
        Device identifier string for the selected camera, or None if skipped.
    """
    _check_pyav()

    cameras = list_cameras()

    print("\n" + "=" * 50)
    print("VIDEO DEVICES (Cameras)")
    print("=" * 50)

    if not cameras:
        print("  No cameras detected")
        print("  (Camera support requires ffmpeg to be installed)")
        print("-" * 50 + "\n")
        return None

    for cam in cameras:
        print(f"  {cam['index']}: {cam['name']}")

    print("  n: No camera (skip)")
    print("-" * 50)

    while True:
        try:
            choice = input(f"Select CAMERA [0-{len(cameras)-1}] or 'n' to skip: ").strip().lower()
            if choice == "n" or choice == "":
                print("  -> No camera selected")
                print("-" * 50 + "\n")
                return None
            idx = int(choice)
            if 0 <= idx < len(cameras):
                selected = cameras[idx]
                print(f"  -> Selected: {selected['name']}")
                print("-" * 50 + "\n")
                return selected["device"]
            print(f"  Invalid choice, enter 0-{len(cameras)-1} or 'n'")
        except ValueError:
            print("  Please enter a number or 'n'")
