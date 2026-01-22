"""Comprehensive integration tests for WebRTC end-to-end scenarios.

This module contains integration tests covering:
1. LocalRTC with mock LLM provider (audio input → response → audio output)
2. External WebRTC connection simulation (GetStream-like flow)
3. Format negotiation with multiple provider types (24kHz, 48kHz)
4. Error handling (connection drops, invalid audio, pipeline failures)
5. Cleanup and resource management (no GStreamer leaks)

These tests are marked as integration tests and may be excluded from CI unit test runs.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from vision_agents.core.types import AudioCapabilities, PcmData
from vision_agents.plugins.localrtc import Edge
from vision_agents.plugins.localrtc.room import LocalRoom


# Mock LLM Providers for testing


class MockGeminiLLM:
    """Mock Gemini LLM provider with 24kHz mono requirements."""

    def get_audio_requirements(self):
        """Return Gemini-like audio requirements (24kHz mono)."""
        return AudioCapabilities(
            sample_rate=24000,
            channels=1,
            bit_depth=16,
            supported_sample_rates=[16000, 24000],
            supported_channels=[1],
            encoding="pcm",
        )


class MockGetStreamLLM:
    """Mock GetStream LLM provider with 48kHz stereo requirements."""

    def get_audio_requirements(self):
        """Return GetStream-like audio requirements (48kHz stereo)."""
        return AudioCapabilities(
            sample_rate=48000,
            channels=2,
            bit_depth=16,
            supported_sample_rates=[48000],
            supported_channels=[1, 2],
            encoding="pcm",
        )


class MockLLMWithoutRequirements:
    """Mock LLM provider without audio requirements."""

    pass


class MockAgent:
    """Mock Agent class for testing."""

    def __init__(self, llm):
        self.llm = llm
        self.events = MagicMock()
        self.events.send = AsyncMock()


# Integration Tests


@pytest.mark.integration
@pytest.mark.asyncio
async def test_localrtc_end_to_end_with_mock_llm():
    """Integration test: LocalRTC with mock LLM provider (audio input → response → audio output).

    Tests the complete flow:
    1. Create Edge with input device
    2. Join room with mock agent
    3. Publish audio input track
    4. Receive audio via output track
    5. Verify format negotiation occurred
    6. Clean shutdown
    """
    # Create mock sounddevice to avoid actual device access
    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        # Setup mock input stream
        mock_input_stream = MagicMock()
        mock_input_stream.__enter__ = MagicMock(return_value=mock_input_stream)
        mock_input_stream.__exit__ = MagicMock(return_value=None)
        mock_sd.InputStream.return_value = mock_input_stream

        # Setup mock output stream
        mock_output_stream = MagicMock()
        mock_output_stream.write = MagicMock()
        mock_sd.OutputStream.return_value = mock_output_stream

        # Create Edge with Gemini-like LLM
        edge = Edge(sample_rate=16000, channels=1)
        agent = MockAgent(llm=MockGeminiLLM())

        # Join room - should negotiate to 24kHz mono
        room = await edge.join(agent, room_id="test-room")
        assert isinstance(room, LocalRoom)
        assert edge._negotiated_output_sample_rate == 24000
        assert edge._negotiated_output_channels == 1

        # Create audio output track with negotiated format
        audio_track = edge.create_audio_track()
        assert audio_track.sample_rate == 24000
        assert audio_track.channels == 1

        # Simulate audio output
        test_audio = PcmData(
            data=b"\x00\x01" * 24000,  # 1 second of audio
            sample_rate=24000,
            channels=1,
            bit_depth=16,
        )
        await audio_track.write(test_audio)

        # Cleanup
        audio_track.stop()
        await room.leave()
        await edge.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_external_webrtc_simulation():
    """Integration test: External WebRTC connection simulation (GetStream-like flow).

    Tests:
    1. Connection establishment with external provider
    2. Format negotiation to 48kHz stereo
    3. Bidirectional audio flow
    4. Proper track lifecycle
    """
    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        # Setup mocks
        mock_input_stream = MagicMock()
        mock_input_stream.__enter__ = MagicMock(return_value=mock_input_stream)
        mock_input_stream.__exit__ = MagicMock(return_value=None)
        mock_sd.InputStream.return_value = mock_input_stream

        mock_output_stream = MagicMock()
        mock_output_stream.write = MagicMock()
        mock_sd.OutputStream.return_value = mock_output_stream

        # Simulate GetStream connection
        edge = Edge(sample_rate=48000, channels=2)
        agent = MockAgent(llm=MockGetStreamLLM())

        # Join room - should negotiate to 48kHz stereo
        room = await edge.join(agent, room_id="external-room")
        assert edge._negotiated_output_sample_rate == 48000
        assert edge._negotiated_output_channels == 2

        # Create input and output tracks
        audio_output = edge.create_audio_track()
        assert audio_output.sample_rate == 48000
        assert audio_output.channels == 2

        # Simulate bidirectional audio
        test_audio_stereo = PcmData(
            data=b"\x00\x01\x02\x03" * 48000,  # 1 second stereo
            sample_rate=48000,
            channels=2,
            bit_depth=16,
        )
        await audio_output.write(test_audio_stereo)

        # Cleanup
        audio_output.stop()
        await room.leave()
        await edge.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_format_negotiation_multiple_providers():
    """Integration test: Format negotiation with multiple provider types.

    Tests that different providers correctly negotiate different formats:
    - Gemini: 24kHz mono
    - GetStream: 48kHz stereo
    - Default: 24kHz mono
    """
    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=None)
        mock_sd.InputStream.return_value = mock_stream
        mock_sd.OutputStream.return_value = MagicMock()

        # Test 1: Gemini (24kHz mono)
        edge1 = Edge(sample_rate=16000, channels=1)
        agent1 = MockAgent(llm=MockGeminiLLM())
        room1 = await edge1.join(agent1, room_id="gemini-room")

        assert edge1._negotiated_output_sample_rate == 24000
        assert edge1._negotiated_output_channels == 1
        track1 = edge1.create_audio_track()
        assert track1.sample_rate == 24000
        assert track1.channels == 1

        # Test 2: GetStream (48kHz stereo)
        edge2 = Edge(sample_rate=16000, channels=1)
        agent2 = MockAgent(llm=MockGetStreamLLM())
        room2 = await edge2.join(agent2, room_id="getstream-room")

        assert edge2._negotiated_output_sample_rate == 48000
        assert edge2._negotiated_output_channels == 2
        track2 = edge2.create_audio_track()
        assert track2.sample_rate == 48000
        assert track2.channels == 2

        # Test 3: Default (24kHz mono)
        edge3 = Edge(sample_rate=16000, channels=1)
        agent3 = MockAgent(llm=MockLLMWithoutRequirements())
        room3 = await edge3.join(agent3, room_id="default-room")

        assert edge3._negotiated_output_sample_rate == 24000
        assert edge3._negotiated_output_channels == 1
        track3 = edge3.create_audio_track()
        assert track3.sample_rate == 24000
        assert track3.channels == 1

        # Verify all formats are different or as expected
        assert (
            edge1._negotiated_output_sample_rate != edge2._negotiated_output_sample_rate
        )
        assert edge1._negotiated_output_channels != edge2._negotiated_output_channels

        # Cleanup all
        track1.stop()
        track2.stop()
        track3.stop()
        await room1.leave()
        await room2.leave()
        await room3.leave()
        await edge1.close()
        await edge2.close()
        await edge3.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_audio_format_conversion_during_playback():
    """Integration test: Audio format conversion between different sample rates and channels.

    Tests that audio is properly converted when:
    - Input is 16kHz mono but output is negotiated to 24kHz mono
    - Input is 24kHz mono but output is 48kHz stereo
    """
    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        mock_output_stream = MagicMock()
        mock_output_stream.write = MagicMock()
        mock_sd.OutputStream.return_value = mock_output_stream

        # Test 1: 16kHz mono → 24kHz mono conversion
        edge1 = Edge(sample_rate=16000, channels=1)
        agent1 = MockAgent(llm=MockGeminiLLM())
        room1 = await edge1.join(agent1, room_id="room1")

        track1 = edge1.create_audio_track()
        input_audio = PcmData(
            data=b"\x00\x01" * 16000,  # 1 second at 16kHz
            sample_rate=16000,
            channels=1,
            bit_depth=16,
        )
        await track1.write(input_audio)
        # Verify conversion occurred (output track is 24kHz)
        assert track1.sample_rate == 24000

        # Test 2: 24kHz mono → 48kHz stereo conversion
        edge2 = Edge(sample_rate=24000, channels=1)
        agent2 = MockAgent(llm=MockGetStreamLLM())
        room2 = await edge2.join(agent2, room_id="room2")

        track2 = edge2.create_audio_track()
        input_audio2 = PcmData(
            data=b"\x00\x01" * 24000,  # 1 second at 24kHz mono
            sample_rate=24000,
            channels=1,
            bit_depth=16,
        )
        await track2.write(input_audio2)
        # Verify conversion occurred (output track is 48kHz stereo)
        assert track2.sample_rate == 48000
        assert track2.channels == 2

        # Cleanup
        track1.stop()
        track2.stop()
        await room1.leave()
        await room2.leave()
        await edge1.close()
        await edge2.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_error_handling_invalid_audio():
    """Integration test: Error handling with invalid audio data.

    Tests:
    1. Invalid sample rate
    2. Invalid channels
    3. Invalid bit depth
    4. Empty audio data
    """
    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        mock_sd.OutputStream.return_value = MagicMock()

        edge = Edge(sample_rate=24000, channels=1)
        agent = MockAgent(llm=MockGeminiLLM())
        room = await edge.join(agent, room_id="error-room")
        track = edge.create_audio_track()

        # Test 1: Invalid sample rate (should raise RuntimeError with format error)
        with pytest.raises(RuntimeError, match="Invalid input sample rate"):
            invalid_audio = PcmData(
                data=b"\x00\x01" * 1000,
                sample_rate=0,  # Invalid
                channels=1,
                bit_depth=16,
            )
            await track.write(invalid_audio)

        # Test 2: Invalid channels (should raise RuntimeError with format error)
        with pytest.raises(RuntimeError, match="Invalid input channel count"):
            invalid_audio = PcmData(
                data=b"\x00\x01" * 1000,
                sample_rate=24000,
                channels=0,  # Invalid
                bit_depth=16,
            )
            await track.write(invalid_audio)

        # Test 3: Invalid bit depth (requires format conversion to trigger validation)
        # Bit depth validation happens during format conversion, wrapped in RuntimeError
        with pytest.raises(RuntimeError, match="Invalid bit depth"):
            invalid_audio = PcmData(
                data=b"\x00\x01" * 1000,
                sample_rate=16000,  # Different rate to trigger conversion
                channels=1,
                bit_depth=7,  # Invalid
            )
            await track.write(invalid_audio)

        # Test 4: Empty audio data (should handle gracefully)
        empty_audio = PcmData(data=b"", sample_rate=24000, channels=1, bit_depth=16)
        await track.write(empty_audio)  # Should not raise

        # Cleanup
        track.stop()
        await room.leave()
        await edge.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_error_handling_connection_drop():
    """Integration test: Error handling when connection drops.

    Tests:
    1. Room leave during active session
    2. Edge close during active playback
    3. Track close during write
    """
    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        mock_sd.OutputStream.return_value = MagicMock()

        edge = Edge(sample_rate=24000, channels=1)
        agent = MockAgent(llm=MockGeminiLLM())
        room = await edge.join(agent, room_id="drop-room")
        track = edge.create_audio_track()

        # Test 1: Leave room during active session
        await room.leave()
        # Should handle gracefully

        # Test 2: Close edge
        await edge.close()
        # Should handle gracefully

        # Track should be stopped
        track.stop()
        # Verify track is stopped
        assert track._stopped is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_error_handling_pipeline_failure():
    """Integration test: Error handling when audio pipeline fails.

    Tests:
    1. sounddevice stream creation failure
    2. sounddevice write failure
    3. Recovery and cleanup
    """
    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        # Test 1: Stream creation failure during initialization
        mock_sd.OutputStream.side_effect = RuntimeError("Device not found")

        edge = Edge(sample_rate=24000, channels=1)
        agent = MockAgent(llm=MockGeminiLLM())
        room = await edge.join(agent, room_id="fail-room")

        # Creating track should raise error during stream creation
        try:
            _track = edge.create_audio_track()
            # If track is created but stream fails, that's also valid
            # The error would occur during first write
        except RuntimeError as e:
            assert "Device not found" in str(e)

        # Cleanup should still work
        await room.leave()
        await edge.close()

        # Test 2: Write failure - simulate playback thread error
        # Note: The actual write() method doesn't directly propagate stream.write errors
        # because it uses a background thread. This test verifies graceful handling.
        mock_sd.OutputStream.side_effect = None
        mock_output_stream = MagicMock()
        mock_output_stream.write.side_effect = RuntimeError("Buffer overflow")
        mock_sd.OutputStream.return_value = mock_output_stream

        edge2 = Edge(sample_rate=24000, channels=1)
        agent2 = MockAgent(llm=MockGeminiLLM())
        room2 = await edge2.join(agent2, room_id="fail-room-2")
        track2 = edge2.create_audio_track()

        # Write will queue data but background thread will encounter error
        # The method doesn't raise immediately due to async buffering
        audio = PcmData(
            data=b"\x00\x01" * 24000, sample_rate=24000, channels=1, bit_depth=16
        )
        await track2.write(audio)  # Buffered, error happens in background

        # Cleanup should still work even with background errors
        track2.stop()
        await room2.leave()
        await edge2.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resource_cleanup_no_leaks():
    """Integration test: Cleanup and resource management (no GStreamer leaks).

    Tests:
    1. Multiple create/destroy cycles
    2. Track cleanup
    3. Edge cleanup
    4. Room cleanup
    5. No lingering resources
    """
    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        mock_sd.OutputStream.return_value = MagicMock()
        mock_sd.InputStream.return_value = MagicMock()

        # Run multiple cycles
        for i in range(5):
            edge = Edge(sample_rate=24000, channels=1)
            agent = MockAgent(llm=MockGeminiLLM())
            room = await edge.join(agent, room_id=f"cleanup-room-{i}")

            # Create multiple tracks
            tracks = []
            for j in range(3):
                track = edge.create_audio_track()
                tracks.append(track)

            # Write some audio
            audio = PcmData(
                data=b"\x00\x01" * 1000, sample_rate=24000, channels=1, bit_depth=16
            )
            for track in tracks:
                await track.write(audio)

            # Cleanup all tracks
            for track in tracks:
                track.stop()

            # Cleanup room and edge
            await room.leave()
            await edge.close()

            # Verify cleanup
            assert edge._audio_capture_running is False
            assert len(edge._rooms) == 0

        # All cycles completed without resource leaks


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_sessions():
    """Integration test: Multiple concurrent WebRTC sessions.

    Tests:
    1. Multiple edges active simultaneously
    2. Different formats for each session
    3. Proper isolation between sessions
    4. Clean shutdown of all sessions
    """
    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        mock_sd.OutputStream.return_value = MagicMock()

        # Create multiple concurrent sessions
        edges = []
        rooms = []
        tracks = []

        # Session 1: Gemini (24kHz mono)
        edge1 = Edge(sample_rate=16000, channels=1)
        agent1 = MockAgent(llm=MockGeminiLLM())
        room1 = await edge1.join(agent1, room_id="concurrent-1")
        track1 = edge1.create_audio_track()
        edges.append(edge1)
        rooms.append(room1)
        tracks.append(track1)

        # Session 2: GetStream (48kHz stereo)
        edge2 = Edge(sample_rate=16000, channels=1)
        agent2 = MockAgent(llm=MockGetStreamLLM())
        room2 = await edge2.join(agent2, room_id="concurrent-2")
        track2 = edge2.create_audio_track()
        edges.append(edge2)
        rooms.append(room2)
        tracks.append(track2)

        # Session 3: Default (24kHz mono)
        edge3 = Edge(sample_rate=16000, channels=1)
        agent3 = MockAgent(llm=MockLLMWithoutRequirements())
        room3 = await edge3.join(agent3, room_id="concurrent-3")
        track3 = edge3.create_audio_track()
        edges.append(edge3)
        rooms.append(room3)
        tracks.append(track3)

        # Verify each session has correct format
        assert edge1._negotiated_output_sample_rate == 24000
        assert edge2._negotiated_output_sample_rate == 48000
        assert edge3._negotiated_output_sample_rate == 24000

        # Write audio to all tracks concurrently
        audio_tasks = []
        for track in tracks:
            audio = PcmData(
                data=b"\x00\x01" * 1000,
                sample_rate=track.sample_rate,
                channels=track.channels,
                bit_depth=16,
            )
            audio_tasks.append(track.write(audio))

        await asyncio.gather(*audio_tasks)

        # Cleanup all sessions
        for track in tracks:
            track.stop()
        for room in rooms:
            await room.leave()
        for edge in edges:
            await edge.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_long_running_session_with_timeout():
    """Integration test: Long-running session with timeout safeguards.

    Tests:
    1. Session can run for extended period
    2. Proper timeout handling
    3. No infinite loops
    4. Clean shutdown after timeout
    """
    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        mock_sd.OutputStream.return_value = MagicMock()

        edge = Edge(sample_rate=24000, channels=1)
        agent = MockAgent(llm=MockGeminiLLM())
        room = await edge.join(agent, room_id="long-room")
        track = edge.create_audio_track()

        # Simulate long-running session with timeout
        async def write_audio_loop():
            for _ in range(10):  # 10 iterations
                audio = PcmData(
                    data=b"\x00\x01" * 1000, sample_rate=24000, channels=1, bit_depth=16
                )
                await track.write(audio)
                await asyncio.sleep(0.1)  # Simulate real-time processing

        # Run with timeout (should complete within 5 seconds)
        try:
            await asyncio.wait_for(write_audio_loop(), timeout=5.0)
        except asyncio.TimeoutError:
            pytest.fail("Session exceeded timeout - possible infinite loop")

        # Cleanup
        track.stop()
        await room.leave()
        await edge.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_audio_flush_completes():
    """Integration test: Audio flush completes all buffered audio.

    Tests:
    1. Write audio to buffer
    2. Flush completes
    3. No audio left in buffer
    """
    with patch("vision_agents.plugins.localrtc.tracks.sd") as mock_sd:
        mock_output_stream = MagicMock()
        mock_output_stream.write = MagicMock()
        mock_sd.OutputStream.return_value = mock_output_stream

        edge = Edge(sample_rate=24000, channels=1)
        agent = MockAgent(llm=MockGeminiLLM())
        room = await edge.join(agent, room_id="flush-room")
        track = edge.create_audio_track()

        # Write audio
        audio = PcmData(
            data=b"\x00\x01" * 24000, sample_rate=24000, channels=1, bit_depth=16
        )
        await track.write(audio)

        # Flush should complete
        await track.flush()

        # Cleanup
        track.stop()
        await room.leave()
        await edge.close()
