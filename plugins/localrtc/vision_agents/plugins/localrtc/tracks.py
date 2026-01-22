"""Audio and video track implementations for local RTC."""

import time
from typing import Optional, Union

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
