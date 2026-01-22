"""Audio and video track implementations for local RTC."""

import asyncio
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import numpy as np
from vision_agents.core.types import PcmData as CorePcmData

if TYPE_CHECKING:
    from .localedge.config import LocalEdgeConfig

logger = logging.getLogger(__name__)

# Check if development mode is enabled via environment variable
_VISION_AGENT_DEV = os.environ.get("VISION_AGENT_DEV", "").lower() in ("true", "1", "yes")

# Import getstream PcmData for compatibility with Gemini plugin
try:
    from getstream.video.rtc.track_util import PcmData as StreamPcmData
    HAS_STREAM_PCM = True
except ImportError:
    StreamPcmData = None  # type: ignore
    HAS_STREAM_PCM = False

# Type alias for backwards compatibility
PcmData = CorePcmData

try:
    import sounddevice as sd
except ImportError:
    sd = None  # type: ignore

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

# Try to import GStreamer
try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    Gst.init(None)
    GST_AVAILABLE = True
except (ImportError, ValueError):
    GST_AVAILABLE = False
    Gst = None  # type: ignore


class AudioInputTrack:
    """Audio input track that captures from a microphone and converts to PcmData.

    This class captures audio from a microphone device and converts it to the
    core PcmData format for processing by Vision Agents.

    Uses a persistent InputStream with callback-based capture to avoid the
    sounddevice errors that occur when repeatedly creating/destroying streams.

    Attributes:
        device: Device identifier (name, index, or 'default')
        sample_rate: Audio sampling rate in Hz (default: 16000)
        channels: Number of audio channels (default: 1 for mono)
        bit_depth: Bits per sample (default: 16)

    Example:
        >>> track = AudioInputTrack(device='default', sample_rate=16000)
        >>> track.start()
        >>> audio_data = track.capture(duration=1.0)
        >>> print(f"Captured {len(audio_data.data)} bytes")
        >>> track.stop()
    """

    def __init__(
        self,
        device: Union[str, int] = "default",
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        bit_depth: Optional[int] = None,
        buffer_duration: Optional[float] = None,
        config: Optional["LocalEdgeConfig"] = None,
    ) -> None:
        """Initialize the audio input track.

        Args:
            device: Device identifier - can be:
                - 'default': Use system default input device
                - str: Device name to search for
                - int: Device index
            sample_rate: Audio sampling rate in Hz (uses config default if not specified)
            channels: Number of audio channels (uses config default if not specified)
            bit_depth: Bits per sample (uses config default if not specified)
            buffer_duration: Duration of the circular buffer in seconds (uses config default if not specified)
            config: Optional LocalEdgeConfig for default values

        Raises:
            RuntimeError: If sounddevice is not available
            ValueError: If device is not found or invalid
        """
        if sd is None:
            raise RuntimeError(
                "sounddevice is not available. Install it with: pip install sounddevice"
            )

        from .localedge.config import LocalEdgeConfig
        self._config = config if config is not None else LocalEdgeConfig()

        self.sample_rate = sample_rate if sample_rate is not None else self._config.audio.input_sample_rate
        self.channels = channels if channels is not None else self._config.audio.input_channels
        self.bit_depth = bit_depth if bit_depth is not None else self._config.audio.bit_depth
        self._device_index: Optional[int] = None

        # Persistent stream state
        self._stream: Optional[sd.InputStream] = None
        self._buffer_lock = threading.Lock()
        self._buffer: bytearray = bytearray()
        buf_duration = buffer_duration if buffer_duration is not None else self._config.audio.input_buffer_duration
        self._buffer_max_bytes = int(
            buf_duration * self.sample_rate * self.channels * (self.bit_depth // 8)
        )
        self._started = False

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

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Callback for the persistent input stream.

        Args:
            indata: Input audio data as numpy array
            frames: Number of frames
            time_info: Time information from PortAudio
            status: Stream status flags
        """
        if status:
            logger.warning(f"[LOCALRTC] Audio input status: {status}")

        # Convert to bytes and add to buffer
        audio_bytes = indata.tobytes()

        with self._buffer_lock:
            self._buffer.extend(audio_bytes)
            # Trim buffer if it exceeds max size (keep most recent data)
            if len(self._buffer) > self._buffer_max_bytes:
                excess = len(self._buffer) - self._buffer_max_bytes
                del self._buffer[:excess]

    def start(self) -> None:
        """Start the persistent audio input stream.

        This method creates and starts the InputStream. Call this before
        using capture(). The stream will run continuously until stop() is called.
        """
        if self._started:
            return

        dtype = f"int{self.bit_depth}"

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=dtype,
            device=self._device_index,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._started = True

    def stop(self) -> None:
        """Stop the persistent audio input stream and release resources."""
        if not self._started:
            return

        self._started = False

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.error(f"Error closing audio input stream: {e}")
            finally:
                self._stream = None

        with self._buffer_lock:
            self._buffer.clear()

    def capture(self, duration: float) -> PcmData:
        """Capture audio from the microphone for a specified duration.

        Args:
            duration: Duration in seconds to capture audio

        Returns:
            PcmData object containing the captured audio

        Raises:
            RuntimeError: If audio capture fails or stream not started
            ValueError: If duration is invalid
        """
        if duration <= 0:
            raise ValueError(f"Duration must be positive, got {duration}")

        # Auto-start stream if not started
        if not self._started:
            self.start()

        # Calculate required bytes
        bytes_per_sample = self.bit_depth // 8
        required_bytes = int(duration * self.sample_rate * self.channels * bytes_per_sample)

        if _VISION_AGENT_DEV:
            logger.debug(
                f"[AUDIO DEBUG] Capturing audio from input device - "
                f"device_index={self._device_index}, duration={duration}s, "
                f"sample_rate={self.sample_rate}Hz, channels={self.channels}, "
                f"bit_depth={self.bit_depth}, required_bytes={required_bytes}"
            )

        # Wait for enough data to accumulate
        timeout = duration + 0.5  # Give a bit of extra time
        start_time = time.time()

        while True:
            with self._buffer_lock:
                if len(self._buffer) >= required_bytes:
                    # Extract the required audio data
                    audio_bytes = bytes(self._buffer[:required_bytes])
                    del self._buffer[:required_bytes]
                    break

            # Check timeout
            if time.time() - start_time > timeout:
                # Return whatever we have
                with self._buffer_lock:
                    if len(self._buffer) > 0:
                        audio_bytes = bytes(self._buffer)
                        self._buffer.clear()
                        break
                    else:
                        raise RuntimeError(
                            f"Timeout waiting for audio data after {timeout}s"
                        )

            # Brief sleep to avoid busy-waiting
            time.sleep(self._config.audio.loop_sleep_interval)

        timestamp = time.time()

        if _VISION_AGENT_DEV:
            logger.debug(
                f"[AUDIO DEBUG] Captured audio from input device - "
                f"data_size={len(audio_bytes)} bytes, timestamp={timestamp}"
            )

        return PcmData(
            data=audio_bytes,
            sample_rate=self.sample_rate,
            channels=self.channels,
            bit_depth=self.bit_depth,
            timestamp=timestamp,
        )

    async def capture_async(self, duration: float) -> PcmData:
        """Asynchronously capture audio from the microphone.

        This method runs the blocking capture operation in a thread pool
        to avoid blocking the event loop.

        Args:
            duration: Duration in seconds to capture audio

        Returns:
            PcmData object containing the captured audio

        Raises:
            RuntimeError: If audio capture fails
            ValueError: If duration is invalid
        """
        # Run blocking capture in thread pool to avoid blocking event loop
        return await asyncio.to_thread(self.capture, duration)


class AudioOutputTrack:
    """Audio output track that plays PcmData to a speaker.

    This class plays audio to a speaker device from PcmData objects.
    It implements the OutputAudioTrack protocol for Vision Agents.

    Uses a persistent OutputStream with a callback-based approach to avoid
    the PortAudio errors that occur when rapidly creating/destroying streams.

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
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        bit_depth: Optional[int] = None,
        buffer_size_ms: Optional[int] = None,
        config: Optional["LocalEdgeConfig"] = None,
    ) -> None:
        """Initialize the audio output track.

        Args:
            device: Device identifier - can be:
                - 'default': Use system default output device
                - str: Device name to search for
                - int: Device index
            sample_rate: Audio sampling rate in Hz (uses config default if not specified)
            channels: Number of audio channels (uses config default if not specified)
            bit_depth: Bits per sample (uses config default if not specified)
            buffer_size_ms: Size of the audio buffer in milliseconds (uses config default if not specified).
                This buffer accommodates TTS systems that send audio faster than real-time.
                A larger buffer prevents audio dropouts during burst delivery.
            config: Optional LocalEdgeConfig for default values

        Raises:
            RuntimeError: If sounddevice is not available
            ValueError: If device is not found or invalid
        """
        if sd is None:
            raise RuntimeError(
                "sounddevice is not available. Install it with: pip install sounddevice"
            )

        from .localedge.config import LocalEdgeConfig
        self._config = config if config is not None else LocalEdgeConfig()

        self.sample_rate = sample_rate if sample_rate is not None else self._config.audio.output_sample_rate
        self.channels = channels if channels is not None else self._config.audio.output_channels
        self.bit_depth = bit_depth if bit_depth is not None else self._config.audio.bit_depth
        self._device_index: Optional[int] = None
        self._stopped = False

        # Thread-safe buffer for audio data
        self._buffer_lock = threading.Lock()
        self._buffer: bytearray = bytearray()
        buf_size_ms = buffer_size_ms if buffer_size_ms is not None else self._config.audio.output_buffer_size_ms
        self._buffer_size_bytes = int(
            (buf_size_ms / 1000) * self.sample_rate * self.channels * (self.bit_depth // 8)
        )

        # Persistent output stream
        self._stream: Optional[sd.OutputStream] = None
        self._stream_started = False

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

        # Adjust channels based on device capabilities
        max_channels = self._get_device_max_channels()
        if self.channels > max_channels:
            self.channels = max(1, max_channels)
            # Recalculate buffer size with adjusted channels
            self._buffer_size_bytes = int(
                (buf_size_ms / 1000) * self.sample_rate * self.channels * (self.bit_depth // 8)
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

    def _playback_thread(self) -> None:
        """Background thread that plays audio from the buffer.

        Uses blocking writes to the output stream for more reliable playback.
        """
        dtype = f"int{self.bit_depth}"
        bytes_per_sample = self.bit_depth // 8
        # Play in chunks using configured playback duration
        chunk_samples = int(self.sample_rate * self._config.audio.playback_chunk_duration)
        chunk_bytes = chunk_samples * self.channels * bytes_per_sample

        try:
            while not self._stopped:
                # Get data from buffer
                with self._buffer_lock:
                    if len(self._buffer) >= chunk_bytes:
                        data = bytes(self._buffer[:chunk_bytes])
                        del self._buffer[:chunk_bytes]
                    elif len(self._buffer) > 0:
                        # Play what we have
                        data = bytes(self._buffer)
                        self._buffer.clear()
                    else:
                        data = None

                if data:
                    # Convert to numpy and play
                    audio_array = np.frombuffer(data, dtype=dtype)
                    if self.channels > 1:
                        audio_array = audio_array.reshape(-1, self.channels)
                    # Write to stream (blocking)
                    self._stream.write(audio_array)
                else:
                    # No data, sleep briefly
                    time.sleep(self._config.audio.loop_sleep_interval)
        except Exception as e:
            if not self._stopped:
                logger.error(f"Playback thread error: {e}")

    def _get_device_max_channels(self) -> int:
        """Get the maximum number of output channels for the device.

        Returns:
            Maximum number of output channels the device supports
        """
        try:
            if self._device_index is None:
                # Get default output device info
                device_info = sd.query_devices(kind="output")
            else:
                device_info = sd.query_devices(self._device_index)

            if isinstance(device_info, dict):
                max_ch = device_info.get("max_output_channels", 2)
            else:
                max_ch = getattr(device_info, "max_output_channels", 2)

            # Ensure we return an int (handles mocked objects in tests)
            if isinstance(max_ch, int):
                return max_ch
            return 2  # Default to stereo
        except Exception as e:
            logger.warning(f"Error querying device max channels: {e}. Defaulting to stereo.")
            return 2  # Default to stereo if we can't query

    def _ensure_stream_started(self) -> None:
        """Ensure the output stream is created and started.

        Creates a persistent OutputStream with blocking writes and a background
        playback thread. This avoids the PortAudio errors that occur when
        rapidly creating/destroying streams with sd.play().
        """
        if self._stream_started:
            return

        try:
            # Check device capabilities and adjust channels if needed
            max_channels = self._get_device_max_channels()
            actual_channels = min(self.channels, max_channels)

            if actual_channels != self.channels:
                # Update the track's channel count to match device capability
                old_channels = self.channels
                self.channels = actual_channels
                # Recalculate buffer size preserving the original duration
                # buffer_size_bytes = duration_s * sample_rate * channels * bytes_per_sample
                bytes_per_sample = self.bit_depth // 8
                old_duration_s = self._buffer_size_bytes / (self.sample_rate * old_channels * bytes_per_sample)
                self._buffer_size_bytes = int(
                    old_duration_s * self.sample_rate * self.channels * bytes_per_sample
                )

            # Create the output stream without callback (for blocking writes)
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=f"int{self.bit_depth}",
                device=self._device_index,
            )
            self._stream.start()
            self._stream_started = True

            # Start playback thread
            self._playback_thread_handle = threading.Thread(
                target=self._playback_thread, daemon=True
            )
            self._playback_thread_handle.start()
        except Exception:
            # Clean up on failure
            if self._stream is not None:
                try:
                    self._stream.close()
                except Exception as e:
                    logger.error(f"Error closing stream during cleanup: {e}")
                finally:
                    self._stream = None
            self._stream_started = False
            raise

    def _convert_audio(
        self,
        data: bytes,
        from_rate: int,
        to_rate: int,
        from_channels: int,
        to_channels: int,
        bit_depth: int,
    ) -> bytes:
        """Convert audio data: resample and/or change channel count.

        CRITICAL FORMAT CONVERSION IMPLEMENTATION:
        This is the PRIMARY audio format conversion method in Vision Agents.
        It handles all sample rate conversion, channel mixing, and bit depth
        adjustments needed to adapt audio between different formats.

        Common Conversion Scenarios:
            - GetStream (48kHz stereo) -> ASR (16kHz mono)
            - TTS output (24kHz mono) -> Speaker (48kHz stereo)
            - Any format -> Device-specific format

        Conversion Algorithm:
            1. Validation: Verify all format parameters are valid
            2. Byte Conversion: Convert raw bytes to numpy float32 array
            3. Channel Mixing:
               - Stereo -> Mono: Averages both channels
               - Mono -> Stereo: Duplicates mono channel
            4. Sample Rate Conversion: Linear interpolation (np.interp)
            5. Clipping: Prevents overflow by clipping to valid range
            6. Type Conversion: Convert back to target bit depth

        Args:
            data: Raw PCM audio data as bytes
            from_rate: Original sample rate in Hz (e.g., 48000)
            to_rate: Target sample rate in Hz (e.g., 16000)
            from_channels: Original number of channels (1=mono, 2=stereo)
            to_channels: Target number of channels (1=mono, 2=stereo)
            bit_depth: Bits per sample (8, 16, 24, or 32)

        Returns:
            Converted audio data as bytes in target format

        Raises:
            ValueError: If audio format parameters are invalid

        Performance Notes:
            - Linear interpolation is fast but basic quality
            - Suitable for voice applications
            - For music/high-quality, consider using scipy.signal.resample

        See Also:
            - AUDIO_DOCUMENTATION.md: Comprehensive audio format guide
            - AudioOutputTrack.write(): Uses this method for format adaptation
        """
        # Validate input parameters
        if not data:
            raise ValueError("Cannot convert empty audio data")

        if from_rate <= 0 or to_rate <= 0:
            raise ValueError(
                f"Invalid sample rates: from_rate={from_rate}Hz, to_rate={to_rate}Hz. "
                f"Sample rates must be positive."
            )

        if from_channels <= 0 or to_channels <= 0:
            raise ValueError(
                f"Invalid channel counts: from_channels={from_channels}, to_channels={to_channels}. "
                f"Channel counts must be positive."
            )

        if bit_depth not in (8, 16, 24, 32):
            raise ValueError(
                f"Invalid bit depth: {bit_depth}. "
                f"Supported bit depths are: 8, 16, 24, 32."
            )

        # Validate data size matches format
        bytes_per_sample = bit_depth // 8
        expected_sample_count = len(data) // (bytes_per_sample * from_channels)
        expected_bytes = expected_sample_count * bytes_per_sample * from_channels

        if len(data) != expected_bytes:
            logger.warning(
                f"[AUDIO FORMAT WARNING] Audio data size mismatch: "
                f"got {len(data)} bytes, expected {expected_bytes} bytes "
                f"for {from_channels} channels with {bit_depth}-bit samples. "
                f"Truncating to nearest complete frame."
            )
            # Truncate to complete frames
            data = data[:expected_bytes]

        # STEP 1: Convert bytes to numpy array
        # Convert raw PCM bytes to numpy array with appropriate integer type
        # Then cast to float32 for mathematical operations (prevents overflow)
        dtype = f"int{bit_depth}"
        audio_array = np.frombuffer(data, dtype=dtype).astype(np.float32)

        # STEP 2: CHANNEL MIXING - Convert to mono for processing
        # Multi-channel audio is stored interleaved: [L, R, L, R, ...]
        # We reshape to separate channels, then mix to mono
        if from_channels > 1:
            # Reshape: [L, R, L, R, ...] -> [[L, R], [L, R], ...]
            audio_array = audio_array.reshape(-1, from_channels)
            # CRITICAL: Mix stereo to mono by averaging channels
            # This preserves audio balance and prevents clipping
            audio_mono = np.mean(audio_array, axis=1)
        else:
            audio_mono = audio_array.flatten()

        # STEP 3: SAMPLE RATE CONVERSION (Resampling)
        # Uses linear interpolation to change sample rate
        # Example: 48000 Hz -> 16000 Hz reduces data by ~66%
        if from_rate != to_rate:
            num_samples = len(audio_mono)
            # Calculate new length based on sample rate ratio
            # e.g., 1000 samples @ 48kHz -> 333 samples @ 16kHz
            new_length = int(num_samples * to_rate / from_rate)

            # CRITICAL: Linear interpolation for resampling
            # Creates new sample points by interpolating between existing ones
            # Fast but basic quality - suitable for voice applications
            old_indices = np.arange(num_samples)
            new_indices = np.linspace(0, num_samples - 1, new_length)
            audio_mono = np.interp(new_indices, old_indices, audio_mono)

        # STEP 4: CHANNEL EXPANSION - Convert mono to target channels
        if to_channels > 1:
            # CRITICAL: Duplicate mono to create stereo
            # Both channels get identical data (not true stereo, but maintains compatibility)
            resampled = np.column_stack([audio_mono] * to_channels)
        else:
            resampled = audio_mono.reshape(-1, 1)

        # STEP 5: BIT DEPTH CONVERSION with clipping
        # Convert float32 back to target integer type
        # CRITICAL: Clipping prevents overflow/distortion from out-of-range values
        # Signed integer ranges: 8-bit: -128 to 127, 16-bit: -32768 to 32767, etc.
        max_val = 2 ** (bit_depth - 1) - 1
        min_val = -(2 ** (bit_depth - 1))
        resampled = np.clip(resampled, min_val, max_val).astype(dtype)

        return resampled.tobytes()

    async def write(self, data) -> None:
        """Write PCM audio data to the output device.

        This method buffers the audio data and plays it through a persistent
        output stream. If the sample rate differs from the track's configuration,
        it will be automatically resampled.

        Args:
            data: PcmData object containing audio to play. Supports both
                  vision_agents.core.types.PcmData and getstream PcmData.

        Raises:
            RuntimeError: If audio playback fails
            ValueError: If data format is invalid
        """
        if self._stopped:
            raise RuntimeError("AudioOutputTrack has been stopped")

        try:
            # Handle both core PcmData and getstream PcmData types
            if isinstance(data, CorePcmData):
                # Core PcmData has .data attribute
                audio_data = data.data
                sample_rate = data.sample_rate
                channels = data.channels
                bit_depth = data.bit_depth

                if _VISION_AGENT_DEV:
                    logger.debug(
                        f"[AUDIO DEBUG] Ingesting CorePcmData - "
                        f"sample_rate={sample_rate}Hz, channels={channels}, "
                        f"bit_depth={bit_depth}, data_size={len(audio_data)} bytes"
                    )
            elif HAS_STREAM_PCM and isinstance(data, StreamPcmData):
                # GetStream PcmData stores samples as numpy array
                # Use samples directly and convert to bytes
                samples = data.samples
                if _VISION_AGENT_DEV:
                    logger.debug(
                        f"StreamPcmData: samples.shape={samples.shape}, "
                        f"dtype={samples.dtype}, rate={data.sample_rate}, ch={data.channels}"
                    )
                if samples.dtype == np.float32:
                    # Convert float32 [-1, 1] to int16
                    samples = (samples * 32767).astype(np.int16)
                elif samples.dtype != np.int16:
                    samples = samples.astype(np.int16)
                audio_data = samples.tobytes()
                sample_rate = data.sample_rate
                channels = data.channels
                bit_depth = 16

                if _VISION_AGENT_DEV:
                    logger.debug(
                        f"[AUDIO DEBUG] Ingesting StreamPcmData - "
                        f"sample_rate={sample_rate}Hz, channels={channels}, "
                        f"bit_depth={bit_depth}, data_size={len(audio_data)} bytes, "
                        f"samples_shape={data.samples.shape}, dtype={data.samples.dtype}"
                    )
            else:
                raise ValueError(f"Unsupported PcmData type: {type(data)}")

            # Validate and convert sample rate and/or channels if needed
            if sample_rate != self.sample_rate or channels != self.channels:
                if _VISION_AGENT_DEV:
                    logger.debug(
                        f"[AUDIO DEBUG] Format conversion required - "
                        f"from {sample_rate}Hz/{channels}ch to {self.sample_rate}Hz/{self.channels}ch"
                    )

                # Validate format parameters before conversion
                if sample_rate <= 0:
                    raise ValueError(
                        f"Invalid input sample rate: {sample_rate}Hz. "
                        f"Sample rate must be positive."
                    )
                if channels <= 0:
                    raise ValueError(
                        f"Invalid input channel count: {channels}. "
                        f"Channel count must be positive."
                    )
                if self.sample_rate <= 0:
                    raise ValueError(
                        f"Invalid output sample rate: {self.sample_rate}Hz. "
                        f"Output track is misconfigured."
                    )

                audio_data = self._convert_audio(
                    audio_data,
                    from_rate=sample_rate,
                    to_rate=self.sample_rate,
                    from_channels=channels,
                    to_channels=self.channels,
                    bit_depth=bit_depth,
                )

                if _VISION_AGENT_DEV:
                    logger.debug(
                        f"[AUDIO DEBUG] Format conversion completed - "
                        f"output data_size={len(audio_data)} bytes"
                    )
            elif _VISION_AGENT_DEV:
                logger.debug(
                    f"[AUDIO DEBUG] No format conversion needed - "
                    f"format matches output track ({self.sample_rate}Hz/{self.channels}ch)"
                )

            # Ensure the output stream is running
            # Run in thread pool to avoid blocking event loop with synchronous stream operations
            await asyncio.to_thread(self._ensure_stream_started)

            if _VISION_AGENT_DEV:
                logger.debug(
                    f"[AUDIO DEBUG] Writing to output device - "
                    f"device_index={self._device_index}, sample_rate={self.sample_rate}Hz, "
                    f"channels={self.channels}, bit_depth={self.bit_depth}, "
                    f"data_size={len(audio_data)} bytes, "
                    f"buffer_size_before={len(self._buffer)} bytes"
                )

            # Add audio data to the buffer (thread-safe)
            with self._buffer_lock:
                # If buffer is getting too large, remove oldest data to prevent memory issues
                if len(self._buffer) > self._buffer_size_bytes:
                    excess = len(self._buffer) - self._buffer_size_bytes + len(audio_data)
                    if excess > 0:
                        if _VISION_AGENT_DEV:
                            logger.warning(
                                f"[AUDIO DEBUG] Buffer overflow - removing {excess} bytes "
                                f"(buffer_size={len(self._buffer)}, limit={self._buffer_size_bytes})"
                            )
                        del self._buffer[:excess]
                self._buffer.extend(audio_data)

                if _VISION_AGENT_DEV:
                    logger.debug(
                        f"[AUDIO DEBUG] Buffer updated - "
                        f"buffer_size_after={len(self._buffer)} bytes"
                    )

        except Exception as e:
            device_info = (
                f"device {self._device_index}"
                if self._device_index is not None
                else "default device"
            )

            # Provide detailed error message with format information
            error_msg = (
                f"Failed to play audio to {device_info}: {e}\n"
                f"Audio format details:\n"
                f"  - Input: {sample_rate if 'sample_rate' in locals() else 'unknown'}Hz, "
                f"{channels if 'channels' in locals() else 'unknown'} channels, "
                f"{bit_depth if 'bit_depth' in locals() else 'unknown'}-bit\n"
                f"  - Output track: {self.sample_rate}Hz, {self.channels} channels, "
                f"{self.bit_depth}-bit\n"
                f"  - Data type: {type(data).__name__}"
            )

            if _VISION_AGENT_DEV:
                logger.error(f"[AUDIO DEBUG] {error_msg}")

            raise RuntimeError(error_msg)

    async def flush(self) -> None:
        """Flush any buffered audio data and wait for playback to complete.

        This method waits for all queued audio to finish playing by polling
        the buffer until it's empty.
        """
        if self._stopped:
            return

        try:
            # Wait for the buffer to drain
            while True:
                with self._buffer_lock:
                    if len(self._buffer) == 0:
                        break
                await asyncio.sleep(self._config.audio.flush_poll_interval)
        except Exception as e:
            raise RuntimeError(f"Failed to flush audio output: {e}")

    def stop(self) -> None:
        """Stop the audio output track and release resources.

        This method stops the persistent output stream and marks the track as stopped.
        """
        if self._stopped:
            return

        self._stopped = True

        try:
            # Wait for playback thread to finish with timeout
            if hasattr(self, '_playback_thread_handle') and self._playback_thread_handle:
                self._playback_thread_handle.join(timeout=2.0)
                if self._playback_thread_handle.is_alive():
                    logger.warning("Audio playback thread did not terminate within timeout")
        except Exception as e:
            logger.error(f"Error waiting for playback thread: {e}")

        try:
            # Stop and close the persistent stream
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception as e:
            logger.error(f"Error stopping audio output stream: {e}")
        finally:
            self._stream = None
            self._stream_started = False

        try:
            # Clear the buffer
            with self._buffer_lock:
                self._buffer.clear()
        except Exception as e:
            logger.error(f"Error clearing audio buffer: {e}")


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
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        config: Optional["LocalEdgeConfig"] = None,
    ) -> None:
        """Initialize the video input track.

        Args:
            device: Device identifier - can be:
                - int: Device index (e.g., 0 for first camera)
                - str: Device path (e.g., '/dev/video0')
            width: Video frame width in pixels (uses config default if not specified)
            height: Video frame height in pixels (uses config default if not specified)
            fps: Frames per second (uses config default if not specified)
            config: Optional LocalEdgeConfig for default values

        Raises:
            RuntimeError: If cv2 (opencv-python) is not available
            ValueError: If device is invalid or parameters are out of range
        """
        if cv2 is None:
            raise RuntimeError(
                "opencv-python is not available. Install it with: pip install opencv-python"
            )

        from .localedge.config import LocalEdgeConfig
        self._config = config if config is not None else LocalEdgeConfig()

        self.width = width if width is not None else self._config.video.default_width
        self.height = height if height is not None else self._config.video.default_height
        self.fps = fps if fps is not None else self._config.video.default_fps

        if self.width <= 0:
            raise ValueError(f"Width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"Height must be positive, got {self.height}")
        if self.fps <= 0:
            raise ValueError(f"FPS must be positive, got {self.fps}")

        self.device = device
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

            try:
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
            except Exception:
                # If property setting fails, release device before re-raising
                self._capture.release()
                self._capture = None
                raise

        except Exception as e:
            if self._capture is not None:
                try:
                    self._capture.release()
                except Exception as release_error:
                    logger.error(f"Error releasing video capture during cleanup: {release_error}")
                finally:
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
                if self._running:  # Only log if we're still supposed to be running
                    logger.error(f"Error in video capture loop: {e}")
                    time.sleep(self._config.audio.error_retry_delay)
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

        # Wait for capture thread to finish with timeout
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=3.0)
            if self._capture_thread.is_alive():
                logger.warning("Video capture thread did not terminate within timeout")
            self._capture_thread = None

        # Release capture device
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception as e:
                logger.error(f"Error releasing video capture device: {e}")
            finally:
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


class BaseGStreamerTrack(ABC):
    """Base class for GStreamer-based track implementations.

    This abstract base class provides common pipeline initialization, teardown,
    threading, buffer handling, and error handling patterns for all GStreamer tracks.

    Attributes:
        pipeline_str: GStreamer pipeline string
        _pipeline: GStreamer pipeline instance
        _running: Flag indicating if the track is running
        _pipeline_thread: Background thread for pipeline operations
    """

    def __init__(self, pipeline: str) -> None:
        """Initialize the base GStreamer track.

        Args:
            pipeline: GStreamer pipeline string

        Raises:
            RuntimeError: If GStreamer is not available
        """
        if not GST_AVAILABLE:
            raise RuntimeError(
                "GStreamer is not available. Install PyGObject and GStreamer."
            )

        self.pipeline_str = pipeline
        self._pipeline: Optional[Any] = None
        self._running = False
        self._pipeline_thread: Optional[threading.Thread] = None

    @abstractmethod
    def _create_pipeline(self) -> None:
        """Create and configure the GStreamer pipeline.

        Subclasses must implement this method to create their specific pipeline
        configuration with appropriate sources, sinks, and elements.

        Raises:
            RuntimeError: If pipeline creation fails
        """
        pass

    def _start_pipeline(self) -> None:
        """Start the GStreamer pipeline.

        This method sets the pipeline state to PLAYING. It should be called
        after _create_pipeline() and before starting any capture/playback operations.

        Raises:
            RuntimeError: If pipeline is not created or fails to start
        """
        if self._pipeline is None:
            raise RuntimeError("Pipeline not created. Call _create_pipeline() first.")

        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Failed to start GStreamer pipeline")

    def _stop_pipeline(self) -> None:
        """Stop the GStreamer pipeline and release resources.

        This method sets the pipeline state to NULL and cleans up the pipeline
        instance. It handles errors gracefully and can be called multiple times safely.
        """
        if self._pipeline is not None:
            try:
                self._pipeline.set_state(Gst.State.NULL)
            except Exception as e:
                logger.warning(f"Error stopping pipeline: {e}")
            finally:
                self._pipeline = None

    def _wait_for_thread(self, timeout: float = 2.0) -> None:
        """Wait for the pipeline thread to finish.

        Args:
            timeout: Maximum time to wait in seconds (default: 2.0)
        """
        if self._pipeline_thread is not None and self._pipeline_thread.is_alive():
            self._pipeline_thread.join(timeout=timeout)
            self._pipeline_thread = None

    def _handle_pipeline_error(self, error: Exception, operation: str) -> RuntimeError:
        """Handle and format pipeline errors consistently.

        Args:
            error: The exception that occurred
            operation: Description of the operation that failed

        Returns:
            RuntimeError with formatted error message
        """
        error_msg = f"GStreamer pipeline error during {operation}: {error}"
        logger.error(error_msg)
        return RuntimeError(error_msg)


class GStreamerAudioInputTrack(BaseGStreamerTrack):
    """Audio input track using GStreamer pipeline.

    This class captures audio using a custom GStreamer pipeline and converts
    it to the core PcmData format for processing by Vision Agents.

    Attributes:
        pipeline_str: GStreamer pipeline string
        sample_rate: Audio sampling rate in Hz (default: 16000)
        channels: Number of audio channels (default: 1)
        bit_depth: Bits per sample (default: 16)

    Example:
        >>> pipeline = "alsasrc device=hw:0 ! audioconvert ! audioresample"
        >>> track = GStreamerAudioInputTrack(pipeline=pipeline, sample_rate=16000)
        >>> audio_data = track.capture(duration=1.0)
        >>> print(f"Captured {len(audio_data.data)} bytes")
    """

    def __init__(
        self,
        pipeline: str,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        bit_depth: Optional[int] = None,
        config: Optional["LocalEdgeConfig"] = None,
    ) -> None:
        """Initialize the GStreamer audio input track.

        Args:
            pipeline: GStreamer pipeline string for audio input
            sample_rate: Audio sampling rate in Hz (uses config default if not specified)
            channels: Number of audio channels (uses config default if not specified)
            bit_depth: Bits per sample (uses config default if not specified)
            config: Optional LocalEdgeConfig for default values

        Raises:
            RuntimeError: If GStreamer is not available
        """
        super().__init__(pipeline)
        from .localedge.config import LocalEdgeConfig
        self._config = config if config is not None else LocalEdgeConfig()

        self.sample_rate = sample_rate if sample_rate is not None else self._config.audio.input_sample_rate
        self.channels = channels if channels is not None else self._config.audio.input_channels
        self.bit_depth = bit_depth if bit_depth is not None else self._config.audio.bit_depth
        self._appsink: Optional[Any] = None

    def _create_pipeline(self) -> None:
        """Create and configure the GStreamer pipeline."""
        # Build full pipeline with appsink for data capture
        caps = f"audio/x-raw,format=S{self.bit_depth}LE,rate={self.sample_rate},channels={self.channels}"
        sink_name = self._config.gstreamer.appsink_name
        full_pipeline = f"{self.pipeline_str} ! appsink name={sink_name} caps={caps}"

        self._pipeline = Gst.parse_launch(full_pipeline)
        self._appsink = self._pipeline.get_by_name(sink_name)

        if self._appsink is None:
            raise RuntimeError(f"Failed to get appsink '{sink_name}' from GStreamer pipeline")

    def capture(self, duration: float) -> PcmData:
        """Capture audio from the GStreamer pipeline for a specified duration.

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
            if self._pipeline is None:
                self._create_pipeline()

            assert self._pipeline is not None
            assert self._appsink is not None

            # Start pipeline using base class method
            self._start_pipeline()

            # Calculate expected bytes
            bytes_per_sample = self.bit_depth // 8
            expected_bytes = int(duration * self.sample_rate * self.channels * bytes_per_sample)

            # Capture data
            captured_data = bytearray()
            timestamp = time.time()
            timeout = duration + 2.0  # Add buffer time
            start_time = time.time()

            while len(captured_data) < expected_bytes:
                if time.time() - start_time > timeout:
                    raise RuntimeError("Timeout waiting for audio data from GStreamer")

                sample = self._appsink.try_pull_sample(Gst.SECOND)
                if sample:
                    buffer = sample.get_buffer()
                    success, map_info = buffer.map(Gst.MapFlags.READ)
                    if success:
                        captured_data.extend(map_info.data)
                        buffer.unmap(map_info)

            # Stop pipeline using base class method
            self._stop_pipeline()

            # Return exactly the requested duration
            final_data = bytes(captured_data[:expected_bytes])

            return PcmData(
                data=final_data,
                sample_rate=self.sample_rate,
                channels=self.channels,
                bit_depth=self.bit_depth,
                timestamp=timestamp,
            )
        except Exception as e:
            self._stop_pipeline()
            raise self._handle_pipeline_error(e, "audio capture")

    async def capture_async(self, duration: float) -> PcmData:
        """Asynchronously capture audio from the GStreamer pipeline.

        Args:
            duration: Duration in seconds to capture audio

        Returns:
            PcmData object containing the captured audio

        Raises:
            RuntimeError: If audio capture fails
            ValueError: If duration is invalid
        """
        return await asyncio.to_thread(self.capture, duration)


class GStreamerVideoInputTrack(BaseGStreamerTrack):
    """Video input track using GStreamer pipeline.

    This class captures video using a custom GStreamer pipeline and provides
    callback-based frame delivery.

    Attributes:
        pipeline_str: GStreamer pipeline string
        width: Video frame width in pixels (default: 640)
        height: Video frame height in pixels (default: 480)
        fps: Frames per second (default: 30)

    Example:
        >>> def on_frame(frame: np.ndarray, timestamp: float):
        ...     print(f"Received frame at {timestamp}")
        >>> pipeline = "v4l2src device=/dev/video0 ! videoconvert"
        >>> track = GStreamerVideoInputTrack(pipeline=pipeline, width=640, height=480)
        >>> track.start(callback=on_frame)
        >>> # Frames will be delivered via callback
        >>> track.stop()
    """

    def __init__(
        self,
        pipeline: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        config: Optional["LocalEdgeConfig"] = None,
    ) -> None:
        """Initialize the GStreamer video input track.

        Args:
            pipeline: GStreamer pipeline string for video input
            width: Video frame width in pixels (uses config default if not specified)
            height: Video frame height in pixels (uses config default if not specified)
            fps: Frames per second (uses config default if not specified)
            config: Optional LocalEdgeConfig for default values

        Raises:
            RuntimeError: If GStreamer is not available
            ValueError: If parameters are out of range
        """
        super().__init__(pipeline)
        from .localedge.config import LocalEdgeConfig
        self._config = config if config is not None else LocalEdgeConfig()

        self.width = width if width is not None else self._config.video.default_width
        self.height = height if height is not None else self._config.video.default_height
        self.fps = fps if fps is not None else self._config.video.default_fps

        if self.width <= 0:
            raise ValueError(f"Width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"Height must be positive, got {self.height}")
        if self.fps <= 0:
            raise ValueError(f"FPS must be positive, got {self.fps}")

        self._appsink: Optional[Any] = None
        self._callback: Optional[Callable[[np.ndarray, float], None]] = None

    def _create_pipeline(self) -> None:
        """Create and configure the GStreamer pipeline."""
        # Build full pipeline with appsink for frame capture
        video_format = self._config.video.format
        caps = f"video/x-raw,format={video_format},width={self.width},height={self.height},framerate={self.fps}/1"
        sink_name = self._config.gstreamer.appsink_name
        full_pipeline = f"{self.pipeline_str} ! videoscale ! videoconvert ! appsink name={sink_name} caps={caps}"

        self._pipeline = Gst.parse_launch(full_pipeline)
        self._appsink = self._pipeline.get_by_name(sink_name)

        if self._appsink is None:
            raise RuntimeError(f"Failed to get appsink '{sink_name}' from GStreamer pipeline")

        # Configure appsink
        self._appsink.set_property("emit-signals", True)
        self._appsink.set_property("drop", True)
        self._appsink.set_property("max-buffers", self._config.video.max_buffers)

    def _capture_loop(self) -> None:
        """Internal capture loop that reads frames and invokes callback."""
        while self._running and self._appsink:
            try:
                sample = self._appsink.try_pull_sample(Gst.SECOND)
                if sample:
                    buffer = sample.get_buffer()
                    success, map_info = buffer.map(Gst.MapFlags.READ)
                    if success:
                        # Convert buffer to numpy array
                        frame_data = np.frombuffer(map_info.data, dtype=np.uint8)
                        frame = frame_data.reshape((self.height, self.width, 3))
                        buffer.unmap(map_info)

                        # Invoke callback
                        timestamp = time.time()
                        if self._callback is not None:
                            self._callback(frame, timestamp)

            except Exception as e:
                if self._running:
                    logger.error(f"Error in GStreamer video capture loop: {e}")
                    time.sleep(self._config.audio.error_retry_delay)
                else:
                    break

    def start(self, callback: Callable[[np.ndarray, float], None]) -> None:
        """Start capturing video frames.

        Opens the GStreamer pipeline and begins capturing frames. Each captured
        frame is delivered to the provided callback function.

        Args:
            callback: Function to call for each frame. Signature:
                callback(frame: np.ndarray, timestamp: float)
                - frame: Video frame as numpy array (BGR format)
                - timestamp: Timestamp in seconds since epoch

        Raises:
            RuntimeError: If pipeline cannot be created or track is already running
            ValueError: If callback is None
        """
        if self._running:
            raise RuntimeError("GStreamerVideoInputTrack is already running")

        if callback is None:
            raise ValueError("Callback cannot be None")

        self._callback = callback

        # Create pipeline
        self._create_pipeline()

        assert self._pipeline is not None

        # Start pipeline using base class method
        self._start_pipeline()

        # Start capture loop in background thread
        self._running = True
        self._pipeline_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._pipeline_thread.start()

    def stop(self) -> None:
        """Stop capturing video frames and release the pipeline.

        This method stops the capture loop, releases the pipeline,
        and cleans up resources. It can be called multiple times safely.
        """
        if not self._running:
            return

        self._running = False

        # Wait for capture thread to finish using base class method
        self._wait_for_thread()

        # Stop pipeline using base class method
        self._stop_pipeline()

        self._appsink = None
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
            RuntimeError: If pipeline cannot be created or frame capture fails
        """
        was_closed = self._pipeline is None
        if was_closed:
            self._create_pipeline()
            assert self._pipeline is not None
            self._start_pipeline()

        try:
            if self._appsink is None:
                raise RuntimeError("GStreamer pipeline is not ready")

            # Wait for a frame with timeout
            sample = self._appsink.try_pull_sample(5 * Gst.SECOND)
            if not sample:
                raise RuntimeError("Failed to capture frame from GStreamer pipeline")

            buffer = sample.get_buffer()
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if not success:
                raise RuntimeError("Failed to map buffer from GStreamer")

            # Convert buffer to numpy array
            frame_data = np.frombuffer(map_info.data, dtype=np.uint8)
            frame = frame_data.reshape((self.height, self.width, 3))
            buffer.unmap(map_info)

            timestamp = time.time()
            return frame, timestamp

        finally:
            # Stop pipeline if we created it
            if was_closed:
                self._stop_pipeline()
                self._appsink = None


class GStreamerAudioOutputTrack(BaseGStreamerTrack):
    """Audio output track using GStreamer pipeline.

    This class plays audio using a custom GStreamer pipeline from PcmData objects.

    Attributes:
        pipeline_str: GStreamer pipeline string
        sample_rate: Audio sampling rate in Hz (default: 16000)
        channels: Number of audio channels (default: 1)
        bit_depth: Bits per sample (default: 16)

    Example:
        >>> track = GStreamerAudioOutputTrack(pipeline="autoaudiosink", sample_rate=16000)
        >>> pcm_data = PcmData(data=b'...', sample_rate=16000, channels=1)
        >>> await track.write(pcm_data)
        >>> await track.flush()
        >>> track.stop()
    """

    def __init__(
        self,
        pipeline: str,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        bit_depth: Optional[int] = None,
        config: Optional["LocalEdgeConfig"] = None,
    ) -> None:
        """Initialize the GStreamer audio output track.

        Args:
            pipeline: GStreamer pipeline string for audio output (sink)
            sample_rate: Audio sampling rate in Hz (uses config default if not specified)
            channels: Number of audio channels (uses config default if not specified)
            bit_depth: Bits per sample (uses config default if not specified)
            config: Optional LocalEdgeConfig for default values

        Raises:
            RuntimeError: If GStreamer is not available
        """
        super().__init__(pipeline)
        from .localedge.config import LocalEdgeConfig
        self._config = config if config is not None else LocalEdgeConfig()

        self.sample_rate = sample_rate if sample_rate is not None else self._config.audio.output_sample_rate
        self.channels = channels if channels is not None else self._config.audio.output_channels
        self.bit_depth = bit_depth if bit_depth is not None else self._config.audio.bit_depth
        self._appsrc: Optional[Any] = None
        self._stopped = False

    def _create_pipeline(self) -> None:
        """Create and configure the GStreamer pipeline."""
        # Build full pipeline with appsrc for data input
        audio_layout = self._config.gstreamer.audio_layout
        caps = f"audio/x-raw,format=S{self.bit_depth}LE,rate={self.sample_rate},channels={self.channels},layout={audio_layout}"
        src_name = self._config.gstreamer.appsrc_name
        full_pipeline = f"appsrc name={src_name} caps={caps} ! audioconvert ! audioresample ! {self.pipeline_str}"

        self._pipeline = Gst.parse_launch(full_pipeline)
        self._appsrc = self._pipeline.get_by_name(src_name)

        if self._appsrc is None:
            raise RuntimeError(f"Failed to get appsrc '{src_name}' from GStreamer pipeline")

        # Configure appsrc
        self._appsrc.set_property("format", Gst.Format.TIME)
        self._appsrc.set_property("is-live", True)

        # Start pipeline using base class method
        self._start_pipeline()

    async def write(self, data: PcmData) -> None:
        """Write PCM audio data to the output device.

        This method sends the audio data through the GStreamer pipeline.
        If the sample rate differs from the track's configuration, it will
        be automatically resampled by GStreamer.

        Args:
            data: PcmData object containing audio to play

        Raises:
            RuntimeError: If audio playback fails
        """
        if self._stopped:
            raise RuntimeError("GStreamerAudioOutputTrack has been stopped")

        try:
            if self._pipeline is None:
                self._create_pipeline()

            assert self._appsrc is not None

            # Create GStreamer buffer
            buffer = Gst.Buffer.new_wrapped(data.data)

            # Push buffer to pipeline
            ret = self._appsrc.emit("push-buffer", buffer)
            if ret != Gst.FlowReturn.OK:
                raise RuntimeError(f"Failed to push buffer to GStreamer: {ret}")

        except Exception as e:
            raise self._handle_pipeline_error(e, "audio playback")

    async def flush(self) -> None:
        """Flush any buffered audio data and wait for playback to complete."""
        if self._stopped or self._pipeline is None:
            return

        try:
            assert self._appsrc is not None
            # Send end-of-stream to flush the pipeline
            self._appsrc.emit("end-of-stream")
            # Wait a bit for EOS to propagate
            await asyncio.sleep(self._config.audio.eos_wait_time)
        except Exception as e:
            raise RuntimeError(f"Failed to flush GStreamer audio output: {e}")

    def stop(self) -> None:
        """Stop the audio output track and release resources."""
        if self._stopped:
            return

        self._stopped = True

        # Stop pipeline using base class method
        self._stop_pipeline()
        self._appsrc = None
