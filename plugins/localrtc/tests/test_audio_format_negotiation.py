"""Integration tests for audio format negotiation system."""

import pytest
from unittest.mock import Mock, MagicMock
from vision_agents.core.types import AudioCapabilities
from vision_agents.plugins.localrtc import Edge


class MockLLMWithRequirements:
    """Mock LLM provider with audio requirements (Gemini-like)."""

    def get_audio_requirements(self):
        """Return Gemini-like audio requirements."""
        return AudioCapabilities(
            sample_rate=24000,
            channels=1,
            bit_depth=16,
            supported_sample_rates=[16000, 24000],
            supported_channels=[1],
            encoding="pcm"
        )


class MockLLMWithDifferentRequirements:
    """Mock LLM provider with different audio requirements (GetStream-like)."""

    def get_audio_requirements(self):
        """Return GetStream-like audio requirements."""
        return AudioCapabilities(
            sample_rate=48000,
            channels=2,
            bit_depth=16,
            supported_sample_rates=[48000],
            supported_channels=[1, 2],
            encoding="pcm"
        )


class MockLLMWithoutRequirements:
    """Mock LLM provider without audio requirements."""

    pass


class MockAgent:
    """Mock Agent class for testing."""

    def __init__(self, llm):
        self.llm = llm


@pytest.mark.asyncio
async def test_format_negotiation_with_gemini_requirements():
    """Test format negotiation with Gemini-like provider requirements."""
    edge = Edge(sample_rate=16000, channels=1)
    mock_agent = MockAgent(llm=MockLLMWithRequirements())

    # Join should negotiate format
    room = await edge.join(mock_agent, room_id="test-room")

    # Verify negotiated format matches Gemini requirements
    assert edge._negotiated_output_sample_rate == 24000
    assert edge._negotiated_output_channels == 1

    await room.leave()
    await edge.close()


@pytest.mark.asyncio
async def test_format_negotiation_with_getstream_requirements():
    """Test format negotiation with GetStream-like provider requirements."""
    edge = Edge(sample_rate=16000, channels=1)
    mock_agent = MockAgent(llm=MockLLMWithDifferentRequirements())

    # Join should negotiate format
    room = await edge.join(mock_agent, room_id="test-room")

    # Verify negotiated format matches GetStream requirements
    assert edge._negotiated_output_sample_rate == 48000
    assert edge._negotiated_output_channels == 2

    await room.leave()
    await edge.close()


@pytest.mark.asyncio
async def test_format_negotiation_without_requirements():
    """Test format negotiation when provider has no requirements."""
    edge = Edge(sample_rate=16000, channels=1)
    mock_agent = MockAgent(llm=MockLLMWithoutRequirements())

    # Join should use default format
    room = await edge.join(mock_agent, room_id="test-room")

    # Verify default format (24kHz mono for Gemini compatibility)
    assert edge._negotiated_output_sample_rate == 24000
    assert edge._negotiated_output_channels == 1

    await room.leave()
    await edge.close()


@pytest.mark.asyncio
async def test_format_negotiation_without_agent():
    """Test format negotiation when no agent is provided."""
    edge = Edge(sample_rate=16000, channels=1)

    # Join without agent should use default format
    room = await edge.join(room_id="test-room")

    # Verify default format is used
    assert edge._negotiated_output_sample_rate == 24000
    assert edge._negotiated_output_channels == 1

    await room.leave()
    await edge.close()


def test_audio_capabilities_supports_format():
    """Test AudioCapabilities.supports_format method."""
    caps = AudioCapabilities(
        sample_rate=24000,
        channels=1,
        bit_depth=16,
        supported_sample_rates=[16000, 24000, 48000],
        supported_channels=[1, 2]
    )

    # Test supported formats
    assert caps.supports_format(24000, 1) is True
    assert caps.supports_format(16000, 1) is True
    assert caps.supports_format(48000, 2) is True

    # Test unsupported formats
    assert caps.supports_format(8000, 1) is False
    assert caps.supports_format(24000, 4) is False


def test_audio_capabilities_get_closest_sample_rate():
    """Test AudioCapabilities.get_closest_sample_rate method."""
    caps = AudioCapabilities(
        sample_rate=24000,
        channels=1,
        bit_depth=16,
        supported_sample_rates=[16000, 24000, 48000],
        supported_channels=[1]
    )

    # Test exact match
    assert caps.get_closest_sample_rate(24000) == 24000

    # Test closest match
    # 20000 is 4000 away from 16000 and 4000 away from 24000, but min() picks first
    assert caps.get_closest_sample_rate(20000) == 16000
    assert caps.get_closest_sample_rate(12000) == 16000
    assert caps.get_closest_sample_rate(40000) == 48000


def test_audio_capabilities_get_closest_channels():
    """Test AudioCapabilities.get_closest_channels method."""
    caps = AudioCapabilities(
        sample_rate=24000,
        channels=1,
        bit_depth=16,
        supported_sample_rates=[24000],
        supported_channels=[1, 2]
    )

    # Test exact match
    assert caps.get_closest_channels(1) == 1
    assert caps.get_closest_channels(2) == 2

    # Test closest match
    assert caps.get_closest_channels(3) == 2
    assert caps.get_closest_channels(4) == 2


def test_audio_capabilities_post_init():
    """Test AudioCapabilities __post_init__ sets defaults."""
    # Test with no supported_sample_rates/channels
    caps = AudioCapabilities(
        sample_rate=24000,
        channels=1,
        bit_depth=16
    )

    assert caps.supported_sample_rates == [24000]
    assert caps.supported_channels == [1]

    # Test with explicit values
    caps2 = AudioCapabilities(
        sample_rate=24000,
        channels=1,
        bit_depth=16,
        supported_sample_rates=[16000, 24000],
        supported_channels=[1, 2]
    )

    assert caps2.supported_sample_rates == [16000, 24000]
    assert caps2.supported_channels == [1, 2]


@pytest.mark.asyncio
async def test_audio_output_track_uses_negotiated_format():
    """Test that AudioOutputTrack uses negotiated format."""
    edge = Edge(sample_rate=16000, channels=1)
    mock_agent = MockAgent(llm=MockLLMWithDifferentRequirements())

    # Join to negotiate format
    room = await edge.join(mock_agent, room_id="test-room")

    # Create audio output track
    audio_track = edge.create_audio_track()

    # Verify track uses negotiated format
    assert audio_track.sample_rate == 48000
    assert audio_track.channels == 2

    await room.leave()
    await edge.close()


@pytest.mark.asyncio
async def test_format_mismatch_warning_logged(caplog):
    """Test that format mismatch warnings are logged."""
    import logging
    caplog.set_level(logging.WARNING)

    # Create provider that doesn't support 16kHz mono
    class MockLLMWithRestrictedFormats:
        def get_audio_requirements(self):
            return AudioCapabilities(
                sample_rate=48000,
                channels=2,
                bit_depth=16,
                supported_sample_rates=[48000],  # Only 48kHz
                supported_channels=[2],  # Only stereo
                encoding="pcm"
            )

    edge = Edge(sample_rate=16000, channels=1)
    mock_agent = MockAgent(llm=MockLLMWithRestrictedFormats())

    # Join should log warning about format mismatch
    room = await edge.join(mock_agent, room_id="test-room")

    # Check that warning was logged
    assert any("AUDIO FORMAT MISMATCH" in record.message for record in caplog.records)

    await room.leave()
    await edge.close()


@pytest.mark.asyncio
async def test_multiple_providers_different_formats():
    """Test that different providers can use different formats."""
    # Test with Gemini-like provider
    edge1 = Edge(sample_rate=16000, channels=1)
    agent1 = MockAgent(llm=MockLLMWithRequirements())
    room1 = await edge1.join(agent1, room_id="test-room-1")
    assert edge1._negotiated_output_sample_rate == 24000

    # Test with GetStream-like provider
    edge2 = Edge(sample_rate=16000, channels=1)
    agent2 = MockAgent(llm=MockLLMWithDifferentRequirements())
    room2 = await edge2.join(agent2, room_id="test-room-2")
    assert edge2._negotiated_output_sample_rate == 48000

    # Verify they have different formats
    assert edge1._negotiated_output_sample_rate != edge2._negotiated_output_sample_rate

    await room1.leave()
    await room2.leave()
    await edge1.close()
    await edge2.close()
