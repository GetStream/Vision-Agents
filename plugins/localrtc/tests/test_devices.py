"""Tests for device enumeration."""

from unittest.mock import MagicMock, patch

from vision_agents.plugins.localrtc.devices import (
    list_audio_inputs,
    list_audio_outputs,
    list_video_inputs,
)


def test_list_audio_inputs_with_devices():
    """Test list_audio_inputs with available devices."""
    mock_devices = [
        {"name": "Microphone 1", "max_input_channels": 2, "max_output_channels": 0},
        {"name": "Microphone 2", "max_input_channels": 1, "max_output_channels": 0},
        {"name": "Speaker", "max_input_channels": 0, "max_output_channels": 2},
    ]

    with patch("vision_agents.plugins.localrtc.devices.sd") as mock_sd:
        mock_sd.query_devices.return_value = mock_devices
        devices = list_audio_inputs()

        assert len(devices) == 2
        assert devices[0]["name"] == "Microphone 1"
        assert devices[0]["index"] == 0
        assert devices[1]["name"] == "Microphone 2"
        assert devices[1]["index"] == 1


def test_list_audio_inputs_no_devices():
    """Test list_audio_inputs with no devices."""
    with patch("vision_agents.plugins.localrtc.devices.sd") as mock_sd:
        mock_sd.query_devices.return_value = []
        devices = list_audio_inputs()

        assert devices == []


def test_list_audio_inputs_no_sounddevice():
    """Test list_audio_inputs when sounddevice is not available."""
    with patch("vision_agents.plugins.localrtc.devices.sd", None):
        devices = list_audio_inputs()
        assert devices == []


def test_list_audio_inputs_exception():
    """Test list_audio_inputs handles exceptions gracefully."""
    with patch("vision_agents.plugins.localrtc.devices.sd") as mock_sd:
        mock_sd.query_devices.side_effect = Exception("Device error")
        devices = list_audio_inputs()

        assert devices == []


def test_list_audio_inputs_with_device_objects():
    """Test list_audio_inputs with device objects instead of dicts."""
    mock_device1 = MagicMock()
    mock_device1.name = "Microphone Object"
    mock_device1.max_input_channels = 1
    mock_device1.max_output_channels = 0

    mock_device2 = MagicMock()
    mock_device2.name = "Speaker Object"
    mock_device2.max_input_channels = 0
    mock_device2.max_output_channels = 2

    with patch("vision_agents.plugins.localrtc.devices.sd") as mock_sd:
        mock_sd.query_devices.return_value = [mock_device1, mock_device2]
        devices = list_audio_inputs()

        assert len(devices) == 1
        assert devices[0]["name"] == "Microphone Object"
        assert devices[0]["index"] == 0


def test_list_audio_outputs_with_devices():
    """Test list_audio_outputs with available devices."""
    mock_devices = [
        {"name": "Speaker 1", "max_input_channels": 0, "max_output_channels": 2},
        {"name": "Speaker 2", "max_input_channels": 0, "max_output_channels": 2},
        {"name": "Microphone", "max_input_channels": 2, "max_output_channels": 0},
    ]

    with patch("vision_agents.plugins.localrtc.devices.sd") as mock_sd:
        mock_sd.query_devices.return_value = mock_devices
        devices = list_audio_outputs()

        assert len(devices) == 2
        assert devices[0]["name"] == "Speaker 1"
        assert devices[0]["index"] == 0
        assert devices[1]["name"] == "Speaker 2"
        assert devices[1]["index"] == 1


def test_list_audio_outputs_no_devices():
    """Test list_audio_outputs with no devices."""
    with patch("vision_agents.plugins.localrtc.devices.sd") as mock_sd:
        mock_sd.query_devices.return_value = []
        devices = list_audio_outputs()

        assert devices == []


def test_list_audio_outputs_no_sounddevice():
    """Test list_audio_outputs when sounddevice is not available."""
    with patch("vision_agents.plugins.localrtc.devices.sd", None):
        devices = list_audio_outputs()
        assert devices == []


def test_list_audio_outputs_exception():
    """Test list_audio_outputs handles exceptions gracefully."""
    with patch("vision_agents.plugins.localrtc.devices.sd") as mock_sd:
        mock_sd.query_devices.side_effect = Exception("Device error")
        devices = list_audio_outputs()

        assert devices == []


def test_list_audio_outputs_with_device_objects():
    """Test list_audio_outputs with device objects instead of dicts."""
    mock_device1 = MagicMock()
    mock_device1.name = "Speaker Object"
    mock_device1.max_input_channels = 0
    mock_device1.max_output_channels = 2

    mock_device2 = MagicMock()
    mock_device2.name = "Microphone Object"
    mock_device2.max_input_channels = 1
    mock_device2.max_output_channels = 0

    with patch("vision_agents.plugins.localrtc.devices.sd") as mock_sd:
        mock_sd.query_devices.return_value = [mock_device1, mock_device2]
        devices = list_audio_outputs()

        assert len(devices) == 1
        assert devices[0]["name"] == "Speaker Object"
        assert devices[0]["index"] == 0


def test_list_video_inputs():
    """Test list_video_inputs returns empty list."""
    devices = list_video_inputs()
    assert devices == []


def test_list_video_inputs_returns_device_info_type():
    """Test list_video_inputs returns correct type."""
    devices = list_video_inputs()
    assert isinstance(devices, list)
