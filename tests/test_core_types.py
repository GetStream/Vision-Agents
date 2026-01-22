"""Unit tests for core type definitions."""

import json
from dataclasses import asdict

import pytest

from vision_agents.core.types import PcmData, TrackType


class TestTrackType:
    """Test suite for TrackType enum."""

    def test_track_type_values(self):
        """Test that TrackType enum has correct string values."""
        assert TrackType.AUDIO == "audio"
        assert TrackType.VIDEO == "video"
        assert TrackType.SCREENSHARE == "screenshare"

    def test_track_type_is_string_enum(self):
        """Test that TrackType values are strings."""
        assert isinstance(TrackType.AUDIO.value, str)
        assert isinstance(TrackType.VIDEO.value, str)
        assert isinstance(TrackType.SCREENSHARE.value, str)

    def test_track_type_equality(self):
        """Test TrackType equality comparisons."""
        assert TrackType.AUDIO == TrackType.AUDIO
        assert TrackType.VIDEO == TrackType.VIDEO
        assert TrackType.SCREENSHARE == TrackType.SCREENSHARE
        assert TrackType.AUDIO != TrackType.VIDEO

    def test_track_type_string_comparison(self):
        """Test TrackType can be compared with strings."""
        assert TrackType.AUDIO == "audio"
        assert TrackType.VIDEO == "video"
        assert TrackType.SCREENSHARE == "screenshare"

    def test_track_type_membership(self):
        """Test all expected track types are present."""
        track_types = list(TrackType)
        assert TrackType.AUDIO in track_types
        assert TrackType.VIDEO in track_types
        assert TrackType.SCREENSHARE in track_types
        assert len(track_types) == 3

    def test_track_type_from_string(self):
        """Test creating TrackType from string values."""
        assert TrackType("audio") == TrackType.AUDIO
        assert TrackType("video") == TrackType.VIDEO
        assert TrackType("screenshare") == TrackType.SCREENSHARE

    def test_track_type_invalid_value(self):
        """Test that invalid values raise ValueError."""
        with pytest.raises(ValueError):
            TrackType("invalid")


class TestPcmData:
    """Test suite for PcmData dataclass."""

    def test_pcm_data_creation_minimal(self):
        """Test creating PcmData with minimal required fields."""
        data = b"\x00\x01\x02\x03"
        pcm = PcmData(data=data, sample_rate=16000, channels=1)

        assert pcm.data == data
        assert pcm.sample_rate == 16000
        assert pcm.channels == 1
        assert pcm.bit_depth == 16  # Default value
        assert pcm.timestamp is None  # Default value

    def test_pcm_data_creation_all_fields(self):
        """Test creating PcmData with all fields including optional ones."""
        data = b"\x00\x01\x02\x03"
        timestamp = 1234567890.123
        pcm = PcmData(
            data=data,
            sample_rate=24000,
            channels=2,
            bit_depth=24,
            timestamp=timestamp,
        )

        assert pcm.data == data
        assert pcm.sample_rate == 24000
        assert pcm.channels == 2
        assert pcm.bit_depth == 24
        assert pcm.timestamp == timestamp

    def test_pcm_data_mono_audio(self):
        """Test PcmData with mono audio (1 channel)."""
        pcm = PcmData(data=b"\x00\x01", sample_rate=16000, channels=1)
        assert pcm.channels == 1

    def test_pcm_data_stereo_audio(self):
        """Test PcmData with stereo audio (2 channels)."""
        pcm = PcmData(data=b"\x00\x01\x02\x03", sample_rate=48000, channels=2)
        assert pcm.channels == 2

    def test_pcm_data_various_sample_rates(self):
        """Test PcmData with various common sample rates."""
        sample_rates = [8000, 16000, 24000, 44100, 48000]
        for rate in sample_rates:
            pcm = PcmData(data=b"\x00\x01", sample_rate=rate, channels=1)
            assert pcm.sample_rate == rate

    def test_pcm_data_various_bit_depths(self):
        """Test PcmData with various bit depths."""
        bit_depths = [8, 16, 24, 32]
        for depth in bit_depths:
            pcm = PcmData(
                data=b"\x00\x01", sample_rate=16000, channels=1, bit_depth=depth
            )
            assert pcm.bit_depth == depth

    def test_pcm_data_with_timestamp(self):
        """Test PcmData with timestamp field."""
        timestamp = 1234567890.123456
        pcm = PcmData(
            data=b"\x00\x01", sample_rate=16000, channels=1, timestamp=timestamp
        )
        assert pcm.timestamp == timestamp

    def test_pcm_data_without_timestamp(self):
        """Test PcmData without timestamp field."""
        pcm = PcmData(data=b"\x00\x01", sample_rate=16000, channels=1)
        assert pcm.timestamp is None

    def test_pcm_data_empty_bytes(self):
        """Test PcmData with empty byte data."""
        pcm = PcmData(data=b"", sample_rate=16000, channels=1)
        assert pcm.data == b""
        assert len(pcm.data) == 0

    def test_pcm_data_large_bytes(self):
        """Test PcmData with large byte data."""
        large_data = b"\x00" * 100000
        pcm = PcmData(data=large_data, sample_rate=16000, channels=1)
        assert pcm.data == large_data
        assert len(pcm.data) == 100000

    def test_pcm_data_equality(self):
        """Test PcmData equality comparison."""
        pcm1 = PcmData(
            data=b"\x00\x01", sample_rate=16000, channels=1, bit_depth=16
        )
        pcm2 = PcmData(
            data=b"\x00\x01", sample_rate=16000, channels=1, bit_depth=16
        )
        assert pcm1 == pcm2

    def test_pcm_data_inequality(self):
        """Test PcmData inequality with different data."""
        pcm1 = PcmData(data=b"\x00\x01", sample_rate=16000, channels=1)
        pcm2 = PcmData(data=b"\x02\x03", sample_rate=16000, channels=1)
        assert pcm1 != pcm2

    def test_pcm_data_inequality_sample_rate(self):
        """Test PcmData inequality with different sample rates."""
        pcm1 = PcmData(data=b"\x00\x01", sample_rate=16000, channels=1)
        pcm2 = PcmData(data=b"\x00\x01", sample_rate=24000, channels=1)
        assert pcm1 != pcm2

    def test_pcm_data_to_dict(self):
        """Test converting PcmData to dictionary using asdict."""
        data = b"\x00\x01\x02\x03"
        timestamp = 1234567890.123
        pcm = PcmData(
            data=data,
            sample_rate=16000,
            channels=1,
            bit_depth=16,
            timestamp=timestamp,
        )

        pcm_dict = asdict(pcm)
        assert pcm_dict["data"] == data
        assert pcm_dict["sample_rate"] == 16000
        assert pcm_dict["channels"] == 1
        assert pcm_dict["bit_depth"] == 16
        assert pcm_dict["timestamp"] == timestamp

    def test_pcm_data_serialization_with_json_compatible_dict(self):
        """Test PcmData can be converted to JSON-compatible format."""
        data = b"\x00\x01\x02\x03"
        timestamp = 1234567890.123
        pcm = PcmData(
            data=data,
            sample_rate=16000,
            channels=1,
            bit_depth=16,
            timestamp=timestamp,
        )

        # Convert to dict and make it JSON-compatible
        pcm_dict = asdict(pcm)
        pcm_dict["data"] = list(pcm_dict["data"])  # bytes to list for JSON

        # Should be serializable to JSON
        json_str = json.dumps(pcm_dict)
        assert isinstance(json_str, str)

        # Should be deserializable
        decoded = json.loads(json_str)
        assert decoded["sample_rate"] == 16000
        assert decoded["channels"] == 1
        assert decoded["bit_depth"] == 16
        assert decoded["timestamp"] == timestamp
        assert decoded["data"] == [0, 1, 2, 3]

    def test_pcm_data_deserialization_from_dict(self):
        """Test creating PcmData from dictionary."""
        data = b"\x00\x01\x02\x03"
        sample_rate = 16000
        channels = 1
        bit_depth = 16
        timestamp = 1234567890.123

        pcm = PcmData(
            data=data,
            sample_rate=sample_rate,
            channels=channels,
            bit_depth=bit_depth,
            timestamp=timestamp,
        )
        assert pcm.data == data
        assert pcm.sample_rate == sample_rate
        assert pcm.channels == channels
        assert pcm.bit_depth == bit_depth
        assert pcm.timestamp == timestamp

    def test_pcm_data_roundtrip_serialization(self):
        """Test PcmData roundtrip: object -> dict -> object."""
        original = PcmData(
            data=b"\x00\x01\x02\x03",
            sample_rate=24000,
            channels=2,
            bit_depth=24,
            timestamp=1234567890.123,
        )

        # Convert to dict
        pcm_dict = asdict(original)

        # Create new instance from dict
        restored = PcmData(**pcm_dict)

        # Should be equal
        assert restored == original
        assert restored.data == original.data
        assert restored.sample_rate == original.sample_rate
        assert restored.channels == original.channels
        assert restored.bit_depth == original.bit_depth
        assert restored.timestamp == original.timestamp

    def test_pcm_data_is_frozen_false(self):
        """Test that PcmData fields can be modified (not frozen)."""
        pcm = PcmData(data=b"\x00\x01", sample_rate=16000, channels=1)

        # Should be able to modify fields
        pcm.data = b"\x02\x03"
        pcm.sample_rate = 24000
        pcm.channels = 2
        pcm.bit_depth = 24
        pcm.timestamp = 1234567890.123

        assert pcm.data == b"\x02\x03"
        assert pcm.sample_rate == 24000
        assert pcm.channels == 2
        assert pcm.bit_depth == 24
        assert pcm.timestamp == 1234567890.123

    def test_pcm_data_type_annotations(self):
        """Test PcmData has proper type annotations."""
        assert PcmData.__annotations__["data"] is bytes
        assert PcmData.__annotations__["sample_rate"] is int
        assert PcmData.__annotations__["channels"] is int
        assert PcmData.__annotations__["bit_depth"] is int
        # Optional[float] is represented as a Union in Python
        assert "timestamp" in PcmData.__annotations__
