"""Device Discovery Example.

This example demonstrates how to discover and select audio/video devices
for use with Local RTC. It shows several methods for device selection:

1. Listing all available devices
2. Selecting a device by index
3. Selecting a device by name

Key Features:
- Device enumeration for audio inputs, outputs, and video inputs
- Flexible device selection methods
- Helper functions for finding devices by name or index

Use Cases:
- Device configuration and testing
- Multi-device setups
- User-facing device selection interfaces
- Troubleshooting device availability
"""

from typing import List, Optional

from vision_agents.plugins import localrtc
from vision_agents.plugins.localrtc.devices import DeviceInfo


def print_devices() -> None:
    """Print all available audio and video devices.

    This function calls Edge.list_devices() and displays the results
    in a user-friendly format, showing device indices and names.

    Example Output:
        Available Audio Input Devices:
          [0] MacBook Pro Microphone
          [1] External USB Microphone

        Available Audio Output Devices:
          [0] MacBook Pro Speakers
          [1] HDMI Audio Output

        Available Video Input Devices:
          [0] FaceTime HD Camera
          [1] USB Webcam
    """
    print("\n" + "=" * 60)
    print("DEVICE DISCOVERY")
    print("=" * 60)

    # Call the static method to list all devices
    devices = localrtc.Edge.list_devices()

    # Display audio input devices
    print("\nAvailable Audio Input Devices:")
    if devices["audio_inputs"]:
        for device in devices["audio_inputs"]:
            print(f"  [{device['index']}] {device['name']}")
    else:
        print("  No audio input devices found")

    # Display audio output devices
    print("\nAvailable Audio Output Devices:")
    if devices["audio_outputs"]:
        for device in devices["audio_outputs"]:
            print(f"  [{device['index']}] {device['name']}")
    else:
        print("  No audio output devices found")

    # Display video input devices
    print("\nAvailable Video Input Devices:")
    if devices["video_inputs"]:
        for device in devices["video_inputs"]:
            print(f"  [{device['index']}] {device['name']}")
    else:
        print("  No video input devices found")

    print("\n" + "=" * 60 + "\n")


def select_device_by_index(
    device_list: List[DeviceInfo], index: int
) -> Optional[DeviceInfo]:
    """Select a device by its index from a device list.

    This function demonstrates how to select a device using its numeric index.
    Device indices are assigned by the system and may change between sessions.

    Args:
        device_list: List of devices (from Edge.list_devices())
        index: The device index to select

    Returns:
        DeviceInfo dictionary if found, None otherwise

    Example:
        >>> devices = localrtc.Edge.list_devices()
        >>> microphone = select_device_by_index(devices["audio_inputs"], 0)
        >>> if microphone:
        ...     print(f"Selected: {microphone['name']}")
    """
    for device in device_list:
        if device["index"] == index:
            return device
    return None


def select_device_by_name(
    device_list: List[DeviceInfo], name: str, partial_match: bool = True
) -> Optional[DeviceInfo]:
    """Select a device by its name from a device list.

    This function demonstrates how to select a device using its name.
    Device names are more stable than indices but may vary by platform.

    Args:
        device_list: List of devices (from Edge.list_devices())
        name: The device name or partial name to search for
        partial_match: If True, match if name is contained in device name (case-insensitive)
                      If False, require exact match (case-insensitive)

    Returns:
        DeviceInfo dictionary if found, None otherwise

    Example:
        >>> devices = localrtc.Edge.list_devices()
        >>> # Exact match
        >>> mic = select_device_by_name(devices["audio_inputs"], "MacBook Pro Microphone", False)
        >>> # Partial match
        >>> mic = select_device_by_name(devices["audio_inputs"], "MacBook")
        >>> if mic:
        ...     print(f"Selected: {mic['name']} (index: {mic['index']})")
    """
    name_lower = name.lower()
    for device in device_list:
        device_name_lower = device["name"].lower()
        if partial_match:
            if name_lower in device_name_lower:
                return device
        else:
            if name_lower == device_name_lower:
                return device
    return None


def demonstrate_device_selection() -> None:
    """Demonstrate various device selection methods.

    This function shows practical examples of:
    1. Selecting a device by index
    2. Selecting a device by exact name
    3. Selecting a device by partial name match
    """
    print("\n" + "=" * 60)
    print("DEVICE SELECTION EXAMPLES")
    print("=" * 60)

    # Get all available devices
    devices = localrtc.Edge.list_devices()

    # Example 1: Select device by index
    print("\n1. Selecting Audio Input by Index:")
    if devices["audio_inputs"]:
        # Try to select the first audio input device (index from the device)
        first_device = devices["audio_inputs"][0]
        selected = select_device_by_index(devices["audio_inputs"], first_device["index"])
        if selected:
            print(f"   ✓ Selected device at index {selected['index']}: {selected['name']}")
        else:
            print(f"   ✗ Device with index {first_device['index']} not found")
    else:
        print("   No audio input devices available")

    # Example 2: Select device by partial name match
    print("\n2. Selecting Audio Output by Name (partial match):")
    if devices["audio_outputs"]:
        # Try to find a device with a common name pattern
        # This will match any device containing these keywords
        search_terms = ["speaker", "headphone", "output", "default"]
        selected = None
        for term in search_terms:
            selected = select_device_by_name(devices["audio_outputs"], term, partial_match=True)
            if selected:
                print(f"   ✓ Found device matching '{term}': {selected['name']} (index: {selected['index']})")
                break
        if not selected:
            # Fall back to first device
            first_device = devices["audio_outputs"][0]
            print(f"   ℹ Using first available device: {first_device['name']} (index: {first_device['index']})")
    else:
        print("   No audio output devices available")

    # Example 3: Select video device by index
    print("\n3. Selecting Video Input by Index:")
    if devices["video_inputs"]:
        first_device = devices["video_inputs"][0]
        selected = select_device_by_index(devices["video_inputs"], first_device["index"])
        if selected:
            print(f"   ✓ Selected video device: {selected['name']} (index: {selected['index']})")
    else:
        print("   No video input devices available")

    # Example 4: Demonstrate exact name matching
    print("\n4. Selecting by Exact Name (case-insensitive):")
    if devices["audio_inputs"]:
        first_device = devices["audio_inputs"][0]
        # Try exact match with the actual device name
        selected = select_device_by_name(
            devices["audio_inputs"], first_device["name"], partial_match=False
        )
        if selected:
            print(f"   ✓ Exact match found: {selected['name']}")
        else:
            print(f"   ✗ Exact match failed for: {first_device['name']}")
    else:
        print("   No audio input devices available")

    print("\n" + "=" * 60 + "\n")


def demonstrate_edge_creation() -> None:
    """Demonstrate creating an Edge instance with discovered devices.

    This function shows how to use device discovery results to configure
    an Edge instance with specific devices.
    """
    print("\n" + "=" * 60)
    print("EDGE CREATION WITH SELECTED DEVICES")
    print("=" * 60)

    devices = localrtc.Edge.list_devices()

    print("\nExample Edge configurations:")
    print("\n# Option 1: Use default devices (recommended for most cases)")
    print('edge = localrtc.Edge(')
    print('    audio_device="default",')
    print('    video_device=0,')
    print('    speaker_device="default",')
    print(')')

    if devices["audio_inputs"] and devices["audio_outputs"]:
        audio_in = devices["audio_inputs"][0]
        audio_out = devices["audio_outputs"][0]
        print("\n# Option 2: Use specific devices by index")
        print('edge = localrtc.Edge(')
        print(f'    audio_device={audio_in["index"]},  # {audio_in["name"]}')
        print('    video_device=0,')
        print(f'    speaker_device={audio_out["index"]},  # {audio_out["name"]}')
        print(')')

        print("\n# Option 3: Use device by name (requires custom selection logic)")
        print('devices = localrtc.Edge.list_devices()')
        print('mic = select_device_by_name(devices["audio_inputs"], "microphone")')
        print('if mic:')
        print('    edge = localrtc.Edge(')
        print('        audio_device=mic["index"],')
        print('        video_device=0,')
        print('        speaker_device="default",')
        print('    )')

    print("\n" + "=" * 60 + "\n")


def main() -> None:
    """Main function to run all device discovery examples.

    This function demonstrates the complete workflow for device discovery
    and selection, which is useful when developing applications that need
    to work with specific audio/video devices.
    """
    print("\n" + "=" * 70)
    print(" " * 15 + "LOCAL RTC DEVICE DISCOVERY")
    print("=" * 70)
    print("\nThis example shows how to discover and select devices for Local RTC.")
    print("Device discovery is useful for:")
    print("  • Configuring specific microphones, cameras, or speakers")
    print("  • Building device selection UIs")
    print("  • Troubleshooting device availability issues")
    print("  • Multi-device setups")
    print("\n" + "=" * 70)

    # Run all examples
    print_devices()
    demonstrate_device_selection()
    demonstrate_edge_creation()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Use Edge.list_devices() to discover all available devices")
    print("  2. Devices have 'name' and 'index' properties")
    print("  3. Select by index for direct access")
    print("  4. Select by name for more user-friendly configuration")
    print("  5. Use 'default' for audio_device/speaker_device for automatic selection")
    print("  6. Video device typically uses numeric index (0 for first camera)")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    """Run the device discovery example.

    Usage:
        python device_discovery.py

    This example does not require any credentials or environment variables.
    It only enumerates local devices and does not create any network connections.

    Requirements:
        - vision-agents-plugin-localrtc
        - Audio/video devices connected to the system (optional for testing)
    """
    main()
