"""Device enumeration for local audio and video devices."""

from typing import TypedDict

try:
    import sounddevice as sd
except ImportError:
    sd = None  # type: ignore


class DeviceInfo(TypedDict):
    """Device information dictionary."""

    name: str
    index: int


def list_audio_inputs() -> list[DeviceInfo]:
    """List available audio input devices.

    Returns:
        List of dictionaries containing device name and index.
        Returns empty list if sounddevice is not available or no devices found.

    Example:
        >>> devices = list_audio_inputs()
        >>> for device in devices:
        ...     print(f"{device['index']}: {device['name']}")
    """
    if sd is None:
        return []

    try:
        devices = sd.query_devices()
        audio_inputs: list[DeviceInfo] = []

        for idx, device in enumerate(devices):
            # Check if device has input channels
            if isinstance(device, dict):
                max_input_channels = device.get("max_input_channels", 0)
                name = device.get("name", "Unknown")
            else:
                max_input_channels = getattr(device, "max_input_channels", 0)
                name = getattr(device, "name", "Unknown")

            if max_input_channels > 0:
                audio_inputs.append({"name": str(name), "index": idx})

        return audio_inputs
    except Exception:
        # Gracefully handle any sounddevice errors
        return []


def list_audio_outputs() -> list[DeviceInfo]:
    """List available audio output devices.

    Returns:
        List of dictionaries containing device name and index.
        Returns empty list if sounddevice is not available or no devices found.

    Example:
        >>> devices = list_audio_outputs()
        >>> for device in devices:
        ...     print(f"{device['index']}: {device['name']}")
    """
    if sd is None:
        return []

    try:
        devices = sd.query_devices()
        audio_outputs: list[DeviceInfo] = []

        for idx, device in enumerate(devices):
            # Check if device has output channels
            if isinstance(device, dict):
                max_output_channels = device.get("max_output_channels", 0)
                name = device.get("name", "Unknown")
            else:
                max_output_channels = getattr(device, "max_output_channels", 0)
                name = getattr(device, "name", "Unknown")

            if max_output_channels > 0:
                audio_outputs.append({"name": str(name), "index": idx})

        return audio_outputs
    except Exception:
        # Gracefully handle any sounddevice errors
        return []


def list_video_inputs() -> list[DeviceInfo]:
    """List available video input devices.

    Returns:
        List of dictionaries containing device name and index.
        Returns empty list as video enumeration is not yet implemented.

    Note:
        This function currently returns an empty list as video device
        enumeration is not yet implemented. Future implementations may
        use platform-specific APIs or libraries like opencv-python.

    Example:
        >>> devices = list_video_inputs()
        >>> for device in devices:
        ...     print(f"{device['index']}: {device['name']}")
    """
    # Video device enumeration not yet implemented
    # Could use opencv-python or platform-specific APIs in the future
    return []
