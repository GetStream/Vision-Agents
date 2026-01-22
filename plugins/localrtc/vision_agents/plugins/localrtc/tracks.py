"""Audio and video track implementations for local RTC."""

import asyncio
import threading
import time
from typing import Callable, Optional, Union

import numpy as np

from vision_agents.core.types import PcmData

try:
    import sounddevice as sd
except ImportError:
    sd = None  # type: ignore

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore


class AudioInputTrack:
    """Audio input track that captures from a microphone and converts to PcmData.

    This class captures audio from a microphone device and converts it to the
    core PcmData format for processing by Vision Agents.

    Attributes:
        device: Device identifier (name, index, or 'default')
        sample_rate: Audio sampling rate in Hz (default: 16000)
        channels: Number of audio channels (default: 1 for mono)
        bit_depth: Bits per sample (default: 16)

    Example:
        >>> track = AudioInputTrack(device='default', sample_rate=16000)
        >>> audio_data = track.capture(duration=1.0)
        >>> print(f"Captured {len(audio_data.data)} bytes")
    """

    def __init__(
        self,
        device: Union[str, int] = "default",
        sample_rate: int = 16000,
        channels: int = 1,
        bit_depth: int = 16,
    ) -> None:
        """Initialize the audio input track.

        Args:
            device: Device identifier - can be:
                - 'default': Use system default input device
                - str: Device name to search for
                - int: Device index
            sample_rate: Audio sampling rate in Hz (default: 16000)
            channels: Number of audio channels (default: 1 for mono)
            bit_depth: Bits per sample (default: 16)

        Raises:
            RuntimeError: If sounddevice is not available
            ValueError: If device is not found or invalid
        """
        if sd is None:
            raise RuntimeError(
                "sounddevice is not available. Install it with: pip install sounddevice"
            )

        self.sample_rate = sample_rate
        self.channels = channels
        self.bit_depth = bit_depth
        self._device_index: Optional[int] = None

        # Resolve device to index
        if device == "default":
            # Use None for sounddevice to auto-select default
            self._device_index = None
        elif isinstance(device, int):
            # Validate device index
            self._validate_device_index(device)
            self._device_index = device
        elif isinstance(device, str):
            # Search for device by name
            self._device_index = self._find_device_by_name(device)
        else:
            raise ValueError(
                f"Invalid device type: {type(device)}. Expected str, int, or 'default'"
            )

    def _validate_device_index(self, index: int) -> None:
        """Validate that a device index exists and has input channels.

        Args:
            index: Device index to validate

        Raises:
            ValueError: If device index is invalid or has no input channels
        """
        try:
            devices = sd.query_devices()
            if index < 0 or index >= len(devices):
                raise ValueError(
                    f"Device index {index} out of range. Available devices: 0-{len(devices)-1}"
                )

            device = devices[index]
            if isinstance(device, dict):
                max_input_channels = device.get("max_input_channels", 0)
                device_name = device.get("name", "Unknown")
            else:
                max_input_channels = getattr(device, "max_input_channels", 0)
                device_name = getattr(device, "name", "Unknown")

            if max_input_channels == 0:
                raise ValueError(
                    f"Device {index} ({device_name}) has no input channels"
                )
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Error validating device index {index}: {e}")

    def _find_device_by_name(self, name: str) -> int:
        """Find a device index by name.

        Args:
            name: Device name to search for (case-insensitive partial match)

        Returns:
            Device index

        Raises:
            ValueError: If device name is not found or has no input channels
        """
        try:
            devices = sd.query_devices()
            name_lower = name.lower()

            for idx, device in enumerate(devices):
                if isinstance(device, dict):
                    device_name = device.get("name", "")
                    max_input_channels = device.get("max_input_channels", 0)
                else:
                    device_name = getattr(device, "name", "")
                    max_input_channels = getattr(device, "max_input_channels", 0)

                if max_input_channels > 0 and name_lower in device_name.lower():
                    return idx

            raise ValueError(
                f"Audio input device '{name}' not found. "
                "Use list_audio_inputs() to see available devices."
            )
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Error searching for device '{name}': {e}")

    def capture(self, duration: float) -> PcmData:
        """Capture audio from the microphone for a specified duration.

        Args:
            duration: Duration in seconds to capture audio

        Returns:
            PcmData object containing the captured audio

        Raises:
            RuntimeError: If audio capture fails
            ValueError: If duration is invalid
        """
        if duration <= 0:
            raise ValueError(f"Duration must be positive, got {duration}")

        try:
            # Calculate number of frames to capture
            frames = int(duration * self.sample_rate)

            # Capture audio
            timestamp = time.time()
            recording = sd.rec(
                frames=frames,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=f"int{self.bit_depth}",
                device=self._device_index,
            )
            sd.wait()  # Wait for recording to complete

            # Convert numpy array to bytes
            audio_bytes = recording.tobytes()

            return PcmData(
                data=audio_bytes,
                sample_rate=self.sample_rate,
                channels=self.channels,
                bit_depth=self.bit_depth,
                timestamp=timestamp,
            )
        except Exception as e:
            device_info = (
                f"device {self._device_index}"
                if self._device_index is not None
                else "default device"
            )
            raise RuntimeError(
                f"Failed to capture audio from {device_info}: {e}"
            )

    async def capture_async(self, duration: float) -> PcmData:
        """Asynchronously capture audio from the microphone.

        This is a convenience method that wraps the synchronous capture method.
        For non-blocking async operation, consider using asyncio.to_thread.

        Args:
            duration: Duration in seconds to capture audio

        Returns:
            PcmData object containing the captured audio

        Raises:
            RuntimeError: If audio capture fails
            ValueError: If duration is invalid
        """
        # Note: sounddevice.rec is blocking, so this is still blocking
        # For true async, we'd need to use a different approach with callbacks
        return self.capture(duration)


class AudioOutputTrack:
    """Audio output track that plays PcmData to a speaker.

    This class plays audio to a speaker device from PcmData objects.
    It implements the OutputAudioTrack protocol for Vision Agents.

    Attributes:
        device: Device identifier (name, index, or 'default')
        sample_rate: Audio sampling rate in Hz (default: 16000)
        channels: Number of audio channels (default: 1 for mono)
        bit_depth: Bits per sample (default: 16)

    Example:
        >>> track = AudioOutputTrack(device='default', sample_rate=16000)
        >>> pcm_data = PcmData(data=b'...', sample_rate=16000, channels=1)
        >>> await track.write(pcm_data)
        >>> await track.flush()
        >>> track.stop()
    """

    def __init__(
        self,
        device: Union[str, int] = "default",
        sample_rate: int = 16000,
        channels: int = 1,
        bit_depth: int = 16,
    ) -> None:
        """Initialize the audio output track.

        Args:
            device: Device identifier - can be:
                - 'default': Use system default output device
                - str: Device name to search for
                - int: Device index
            sample_rate: Audio sampling rate in Hz (default: 16000)
            channels: Number of audio channels (default: 1 for mono)
            bit_depth: Bits per sample (default: 16)

        Raises:
            RuntimeError: If sounddevice is not available
            ValueError: If device is not found or invalid
        """
        if sd is None:
            raise RuntimeError(
                "sounddevice is not available. Install it with: pip install sounddevice"
            )

        self.sample_rate = sample_rate
        self.channels = channels
        self.bit_depth = bit_depth
        self._device_index: Optional[int] = None
        self._buffer: bytearray = bytearray()
        self._lock = asyncio.Lock()
        self._stream: Optional[sd.OutputStream] = None
        self._stopped = False

        # Resolve device to index
        if device == "default":
            # Use None for sounddevice to auto-select default
            self._device_index = None
        elif isinstance(device, int):
            # Validate device index
            self._validate_device_index(device)
            self._device_index = device
        elif isinstance(device, str):
            # Search for device by name
            self._device_index = self._find_device_by_name(device)
        else:
            raise ValueError(
                f"Invalid device type: {type(device)}. Expected str, int, or 'default'"
            )

    def _validate_device_index(self, index: int) -> None:
        """Validate that a device index exists and has output channels.

        Args:
            index: Device index to validate

        Raises:
            ValueError: If device index is invalid or has no output channels
        """
        try:
            devices = sd.query_devices()
            if index < 0 or index >= len(devices):
                raise ValueError(
                    f"Device index {index} out of range. Available devices: 0-{len(devices)-1}"
                )

            device = devices[index]
            if isinstance(device, dict):
                max_output_channels = device.get("max_output_channels", 0)
                device_name = device.get("name", "Unknown")
            else:
                max_output_channels = getattr(device, "max_output_channels", 0)
                device_name = getattr(device, "name", "Unknown")

            if max_output_channels == 0:
                raise ValueError(
                    f"Device {index} ({device_name}) has no output channels"
                )
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Error validating device index {index}: {e}")

    def _find_device_by_name(self, name: str) -> int:
        """Find a device index by name.

        Args:
            name: Device name to search for (case-insensitive partial match)

        Returns:
            Device index

        Raises:
            ValueError: If device name is not found or has no output channels
        """
        try:
            devices = sd.query_devices()
            name_lower = name.lower()

            for idx, device in enumerate(devices):
                if isinstance(device, dict):
                    device_name = device.get("name", "")
                    max_output_channels = device.get("max_output_channels", 0)
                else:
                    device_name = getattr(device, "name", "")
                    max_output_channels = getattr(device, "max_output_channels", 0)

                if max_output_channels > 0 and name_lower in device_name.lower():
                    return idx

            raise ValueError(
                f"Audio output device '{name}' not found. "
                "Use list_audio_outputs() to see available devices."
            )
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Error searching for device '{name}': {e}")

    def _convert_sample_rate(
        self, data: bytes, from_rate: int, to_rate: int, channels: int, bit_depth: int
    ) -> bytes:
        """Convert audio data from one sample rate to another.

        Args:
            data: Raw PCM audio data
            from_rate: Original sample rate
            to_rate: Target sample rate
            channels: Number of audio channels
            bit_depth: Bits per sample

        Returns:
            Resampled audio data as bytes
        """
        # Convert bytes to numpy array
        dtype = f"int{bit_depth}"
        audio_array = np.frombuffer(data, dtype=dtype)

        # Reshape for multi-channel audio
        if channels > 1:
            audio_array = audio_array.reshape(-1, channels)

        # Calculate resampling ratio
        num_samples = len(audio_array)
        new_length = int(num_samples * to_rate / from_rate)

        # Perform linear interpolation for resampling
        resampled = np.interp(
            np.linspace(0, num_samples - 1, new_length),
            np.arange(num_samples),
            audio_array.flatten() if channels == 1 else audio_array[:, 0],
        )

        # Convert back to original dtype
        resampled = resampled.astype(dtype)

        # For multi-channel, duplicate to all channels (simple approach)
        if channels > 1:
            resampled = np.column_stack([resampled] * channels)

        return resampled.tobytes()

    async def write(self, data: PcmData) -> None:
        """Write PCM audio data to the output device.

        This method buffers the audio data and plays it through the speaker.
        If the sample rate differs from the track's configuration, it will
        be automatically resampled.

        Args:
            data: PcmData object containing audio to play

        Raises:
            RuntimeError: If audio playback fails
            ValueError: If data format is invalid
        """
        if self._stopped:
            raise RuntimeError("AudioOutputTrack has been stopped")

        try:
            audio_data = data.data

            # Convert sample rate if needed
            if data.sample_rate != self.sample_rate:
                audio_data = self._convert_sample_rate(
                    audio_data,
                    data.sample_rate,
                    self.sample_rate,
                    data.channels,
                    data.bit_depth,
                )

            # Convert to numpy array for playback
            dtype = f"int{self.bit_depth}"
            audio_array = np.frombuffer(audio_data, dtype=dtype)

            # Reshape for multi-channel audio
            if self.channels > 1:
                audio_array = audio_array.reshape(-1, self.channels)

            # Play audio (blocking)
            await asyncio.to_thread(
                sd.play,
                audio_array,
                samplerate=self.sample_rate,
                device=self._device_index,
            )

        except Exception as e:
            device_info = (
                f"device {self._device_index}"
                if self._device_index is not None
                else "default device"
            )
            raise RuntimeError(f"Failed to play audio to {device_info}: {e}")

    async def flush(self) -> None:
        """Flush any buffered audio data and wait for playback to complete.

        This method waits for all queued audio to finish playing.
        """
        if self._stopped:
            return

        try:
            # Wait for playback to complete
            await asyncio.to_thread(sd.wait)
        except Exception as e:
            raise RuntimeError(f"Failed to flush audio output: {e}")

    def stop(self) -> None:
        """Stop the audio output track and release resources.

        This method stops any ongoing playback and marks the track as stopped.
        """
        if self._stopped:
            return

        self._stopped = True

        try:
            # Stop any ongoing playback
            sd.stop()
        except Exception:
            # Ignore errors during stop
            pass


class VideoInputTrack:
    """Video input track that captures frames from a camera.

    This class captures video frames from a camera device using OpenCV (cv2).
    It supports device selection by index or path and provides callback-based
    frame delivery.

    Attributes:
        device: Device identifier (index or path)
        width: Video frame width in pixels (default: 640)
        height: Video frame height in pixels (default: 480)
        fps: Frames per second (default: 30)

    Example:
        >>> def on_frame(frame: np.ndarray, timestamp: float):
        ...     print(f"Received frame at {timestamp}")
        >>> track = VideoInputTrack(device=0, width=640, height=480)
        >>> track.start(callback=on_frame)
        >>> # Frames will be delivered via callback
        >>> track.stop()
    """

    def __init__(
        self,
        device: Union[int, str] = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        """Initialize the video input track.

        Args:
            device: Device identifier - can be:
                - int: Device index (e.g., 0 for first camera)
                - str: Device path (e.g., '/dev/video0')
            width: Video frame width in pixels (default: 640)
            height: Video frame height in pixels (default: 480)
            fps: Frames per second (default: 30)

        Raises:
            RuntimeError: If cv2 (opencv-python) is not available
            ValueError: If device is invalid or parameters are out of range
        """
        if cv2 is None:
            raise RuntimeError(
                "opencv-python is not available. Install it with: pip install opencv-python"
            )

        if width <= 0:
            raise ValueError(f"Width must be positive, got {width}")
        if height <= 0:
            raise ValueError(f"Height must be positive, got {height}")
        if fps <= 0:
            raise ValueError(f"FPS must be positive, got {fps}")

        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self._capture: Optional[cv2.VideoCapture] = None
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable[[np.ndarray, float], None]] = None

    def _open_device(self) -> None:
        """Open the video capture device.

        Raises:
            RuntimeError: If device cannot be opened
        """
        try:
            # Open device
            self._capture = cv2.VideoCapture(self.device)

            if not self._capture.isOpened():
                device_str = str(self.device)
                raise RuntimeError(
                    f"Failed to open video device {device_str}. "
                    "Check that the device exists and is not in use."
                )

            # Set capture properties
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._capture.set(cv2.CAP_PROP_FPS, self.fps)

            # Verify actual settings (devices may not support requested values)
            actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self._capture.get(cv2.CAP_PROP_FPS))

            # Update to actual values
            self.width = actual_width
            self.height = actual_height
            # Only update fps if device reported a valid value
            if actual_fps > 0:
                self.fps = actual_fps

        except Exception as e:
            if self._capture is not None:
                self._capture.release()
                self._capture = None
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"Error opening video device {self.device}: {e}")

    def _capture_loop(self) -> None:
        """Internal capture loop that reads frames and invokes callback.

        This runs continuously while the track is started, reading frames
        from the camera and invoking the callback for each frame.
        """
        frame_duration = 1.0 / self.fps
        last_frame_time = time.time()

        while self._running and self._capture is not None:
            try:
                # Calculate time to next frame
                current_time = time.time()
                elapsed = current_time - last_frame_time
                sleep_time = max(0, frame_duration - elapsed)

                if sleep_time > 0:
                    time.sleep(sleep_time)

                # Read frame
                ret, frame = self._capture.read()

                if not ret:
                    # Failed to read frame - device may be disconnected
                    raise RuntimeError("Failed to read frame from video device")

                # Invoke callback
                timestamp = time.time()
                last_frame_time = timestamp

                if self._callback is not None:
                    self._callback(frame, timestamp)

            except Exception as e:
                # Log error but continue capturing
                # In production, you might want to emit an error event
                if self._running:  # Only print if we're still supposed to be running
                    print(f"Error in video capture loop: {e}")
                    time.sleep(0.1)  # Brief pause before retry
                else:
                    # We're shutting down, exit gracefully
                    break

    def start(self, callback: Callable[[np.ndarray, float], None]) -> None:
        """Start capturing video frames.

        Opens the camera device and begins capturing frames. Each captured
        frame is delivered to the provided callback function.

        Args:
            callback: Function to call for each frame. Signature:
                callback(frame: np.ndarray, timestamp: float)
                - frame: Video frame as numpy array (BGR format)
                - timestamp: Timestamp in seconds since epoch

        Raises:
            RuntimeError: If device cannot be opened or track is already running
            ValueError: If callback is None
        """
        if self._running:
            raise RuntimeError("VideoInputTrack is already running")

        if callback is None:
            raise ValueError("Callback cannot be None")

        self._callback = callback

        # Open device
        self._open_device()

        # Start capture loop in background thread
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    def stop(self) -> None:
        """Stop capturing video frames and release the camera device.

        This method stops the capture loop, releases the camera device,
        and cleans up resources. It can be called multiple times safely.
        """
        if not self._running:
            return

        self._running = False

        # Wait for capture thread to finish
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None

        # Release capture device
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                # Ignore errors during release
                pass
            self._capture = None

        self._callback = None

    def capture_frame(self) -> tuple[np.ndarray, float]:
        """Capture a single frame synchronously.

        This is a convenience method for capturing a single frame without
        starting the continuous capture loop.

        Returns:
            Tuple of (frame, timestamp) where:
                - frame: Video frame as numpy array (BGR format)
                - timestamp: Timestamp in seconds since epoch

        Raises:
            RuntimeError: If device cannot be opened or frame capture fails
        """
        # Open device if not already open
        was_closed = self._capture is None
        if was_closed:
            self._open_device()

        try:
            if self._capture is None:
                raise RuntimeError("Video capture device is not open")

            ret, frame = self._capture.read()

            if not ret:
                raise RuntimeError("Failed to read frame from video device")

            timestamp = time.time()
            return frame, timestamp

        finally:
            # Close device if we opened it
            if was_closed and self._capture is not None:
                self._capture.release()
                self._capture = None
