"""Tests for local plugin device enumeration."""

from unittest.mock import MagicMock, patch

from vision_agents.plugins.local.devices import (
    AudioInputDevice,
    AudioOutputDevice,
    list_audio_input_devices,
    list_audio_output_devices,
    list_cameras,
)
from vision_agents.plugins.local.tracks import _get_camera_input_format


class TestListAudioDevices:
    """Tests for audio device enumeration."""

    def test_list_audio_input_devices(self) -> None:
        with patch("vision_agents.plugins.local.devices.sd") as mock_sd:
            mock_sd.query_devices.return_value = [
                {
                    "name": "Mic",
                    "default_samplerate": 44100,
                    "max_input_channels": 2,
                    "max_output_channels": 0,
                },
                {
                    "name": "Speaker",
                    "default_samplerate": 48000,
                    "max_input_channels": 0,
                    "max_output_channels": 2,
                },
            ]
            mock_sd.default.device = (0, 1)

            devices = list_audio_input_devices()

        assert len(devices) == 1
        assert isinstance(devices[0], AudioInputDevice)
        assert devices[0].name == "Mic"
        assert devices[0].channels == 2

    def test_list_audio_output_devices(self) -> None:
        with patch("vision_agents.plugins.local.devices.sd") as mock_sd:
            mock_sd.query_devices.return_value = [
                {
                    "name": "Mic",
                    "default_samplerate": 44100,
                    "max_input_channels": 2,
                    "max_output_channels": 0,
                },
                {
                    "name": "Speaker",
                    "default_samplerate": 48000,
                    "max_input_channels": 0,
                    "max_output_channels": 2,
                },
            ]
            mock_sd.default.device = (0, 1)

            devices = list_audio_output_devices()

        assert len(devices) == 1
        assert isinstance(devices[0], AudioOutputDevice)
        assert devices[0].name == "Speaker"
        assert devices[0].channels == 2


class TestCameraEnumeration:
    """Tests for camera enumeration functions."""

    def test_list_cameras_darwin(self) -> None:
        with (
            patch("vision_agents.plugins.local.devices.subprocess.run") as run_stub,
            patch(
                "vision_agents.plugins.local.devices.platform.system",
                return_value="Darwin",
            ),
        ):
            run_stub.return_value = MagicMock(
                stderr=(
                    "[AVFoundation video devices:]\n"
                    "[AVFoundation indev @ 0x1] [0] FaceTime HD Camera\n"
                    "[AVFoundation indev @ 0x1] [1] USB Webcam\n"
                    "[AVFoundation audio devices:]\n"
                )
            )

            cameras = list_cameras()

        assert len(cameras) == 2
        assert cameras[0].index == 0
        assert cameras[0].name == "FaceTime HD Camera"
        assert cameras[0].device == "0"
        assert cameras[1].index == 1
        assert cameras[1].name == "USB Webcam"

    def test_list_cameras_linux(self) -> None:
        with (
            patch(
                "vision_agents.plugins.local.devices.platform.system",
                return_value="Linux",
            ),
            patch(
                "vision_agents.plugins.local.devices.glob.glob",
                return_value=["/dev/video0"],
            ),
            patch(
                "builtins.open",
                MagicMock(
                    return_value=MagicMock(
                        __enter__=MagicMock(
                            return_value=MagicMock(
                                read=MagicMock(return_value="USB Camera\n")
                            )
                        ),
                        __exit__=MagicMock(return_value=False),
                    )
                ),
            ),
        ):
            cameras = list_cameras()

        assert len(cameras) == 1
        assert cameras[0].name == "USB Camera"
        assert cameras[0].device == "/dev/video0"

    def test_list_cameras_empty_on_failure(self) -> None:
        with (
            patch(
                "vision_agents.plugins.local.devices.platform.system",
                return_value="Darwin",
            ),
            patch(
                "vision_agents.plugins.local.devices.subprocess.run",
                side_effect=FileNotFoundError,
            ),
        ):
            cameras = list_cameras()

        assert cameras == []

    def test_get_camera_input_format_darwin(self) -> None:
        with patch(
            "vision_agents.plugins.local.tracks.platform.system",
            return_value="Darwin",
        ):
            assert _get_camera_input_format() == "avfoundation"

    def test_get_camera_input_format_linux(self) -> None:
        with patch(
            "vision_agents.plugins.local.tracks.platform.system",
            return_value="Linux",
        ):
            assert _get_camera_input_format() == "v4l2"

    def test_get_camera_input_format_windows(self) -> None:
        with patch(
            "vision_agents.plugins.local.tracks.platform.system",
            return_value="Windows",
        ):
            assert _get_camera_input_format() == "dshow"
