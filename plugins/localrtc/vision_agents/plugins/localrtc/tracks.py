"""Audio and video track implementations for local RTC."""

import asyncio
import time
from typing import Optional, Union

import numpy as np

from vision_agents.core.types import PcmData

try:
    import sounddevice as sd
except ImportError:
    sd = None  # type: ignore


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
