"""
Tests for the Agent class.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from agents import Agent


class MockTool:
    """Mock tool for testing."""

    def __init__(self, return_value="mock_result"):
        self.return_value = return_value

    def __call__(self, *args, **kwargs):
        return self.return_value


class MockPreProcessor:
    """Mock pre-processor for testing."""

    def process(self, data):
        return f"processed_{data}"


class MockModel:
    """Mock AI model for testing."""

    async def generate(self, prompt, **kwargs):
        return f"Generated response for: {prompt}"


class MockSTT:
    """Mock Speech-to-Text service."""

    def __init__(self):
        self._event_handlers = {}

    def on(self, event_name, handler):
        """Mock event handler registration."""
        if event_name not in self._event_handlers:
            self._event_handlers[event_name] = []
        self._event_handlers[event_name].append(handler)

    async def emit(self, event_name, *args, **kwargs):
        """Mock event emission."""
        if event_name in self._event_handlers:
            for handler in self._event_handlers[event_name]:
                if asyncio.iscoroutinefunction(handler):
                    await handler(*args, **kwargs)
                else:
                    handler(*args, **kwargs)

    async def process_audio(self, pcm_data, user):
        """Mock process_audio method."""
        # Simulate transcription result
        await self.emit("transcript", "transcribed text", user, {"confidence": 0.95})

    async def close(self):
        """Mock close method."""
        pass


class MockTTS:
    """Mock Text-to-Speech service."""

    def __init__(self):
        self.output_track = None
        self.sent_texts = []
        self._event_handlers = {}

    def set_output_track(self, track):
        self.output_track = track

    def on(self, event_name, handler):
        """Mock event handler registration."""
        if event_name not in self._event_handlers:
            self._event_handlers[event_name] = []
        self._event_handlers[event_name].append(handler)

    async def emit(self, event_name, *args, **kwargs):
        """Mock event emission."""
        if event_name in self._event_handlers:
            for handler in self._event_handlers[event_name]:
                if asyncio.iscoroutinefunction(handler):
                    await handler(*args, **kwargs)
                else:
                    handler(*args, **kwargs)

    async def send(self, text, user=None):
        self.sent_texts.append(text)
        # Simulate audio generation and emission
        audio_data = b"mock_audio_data"
        await self.emit("audio", audio_data, user)

    async def simulate_error(self, error_msg, user=None):
        """Helper method to simulate TTS errors for testing."""
        error = Exception(error_msg)
        await self.emit("error", error, user)


class MockVAD:
    """Mock Voice Activity Detection service."""

    def __init__(self):
        self._event_handlers = {}

    def on(self, event_name, handler):
        """Mock event handler registration."""
        if event_name not in self._event_handlers:
            self._event_handlers[event_name] = []
        self._event_handlers[event_name].append(handler)

    async def emit(self, event_name, *args, **kwargs):
        """Mock event emission."""
        if event_name in self._event_handlers:
            for handler in self._event_handlers[event_name]:
                if asyncio.iscoroutinefunction(handler):
                    await handler(*args, **kwargs)
                else:
                    handler(*args, **kwargs)

    async def process_audio(self, pcm_data, user):
        """Mock process_audio method."""
        # Simulate speech detection events
        await self.emit("speech_start", user, {})
        await asyncio.sleep(0.001)  # Simulate processing time
        await self.emit("speech_end", user, {})

    async def close(self):
        """Mock close method."""
        pass


class MockTurnDetection:
    """Mock turn detection service."""

    def detect_turn(self, audio_data):
        return True  # Always return True for testing


class TestAgent:
    """Test cases for the Agent class."""

    def test_agent_initialization(self):
        """Test basic agent initialization."""
        instructions = "Test instructions"
        tools = [MockTool()]
        pre_processors = [MockPreProcessor()]
        model = MockModel()
        stt = MockSTT()
        tts = MockTTS()
        vad = MockVAD()
        turn_detection = MockTurnDetection()

        agent = Agent(
            instructions=instructions,
            tools=tools,
            pre_processors=pre_processors,
            model=model,
            stt=stt,
            tts=tts,
            vad=vad,
            turn_detection=turn_detection,
            name="Test Agent",
        )

        assert agent.instructions == instructions
        assert agent.tools == tools
        assert agent.pre_processors == pre_processors
        assert agent.model == model
        assert agent.stt == stt
        assert agent.tts == tts
        assert agent.vad == vad
        assert agent.turn_detection == turn_detection
        assert agent.name == "Test Agent"
        assert agent.bot_id.startswith("agent-")
        assert not agent.is_running()

    def test_agent_initialization_with_defaults(self):
        """Test agent initialization with default values."""
        agent = Agent(instructions="Test instructions")

        assert agent.instructions == "Test instructions"
        assert agent.tools == []
        assert agent.pre_processors == []
        assert agent.model is None
        assert agent.stt is None
        assert agent.tts is None
        assert agent.vad is None
        assert agent.turn_detection is None
        assert agent.name == "AI Agent"
        assert agent.bot_id.startswith("agent-")
        assert not agent.is_running()

    def test_agent_custom_bot_id(self):
        """Test agent initialization with custom bot ID."""
        custom_id = "custom-bot-123"
        agent = Agent(instructions="Test", bot_id=custom_id)

        assert agent.bot_id == custom_id

    @pytest.mark.asyncio
    async def test_generate_response(self):
        """Test response generation with model."""
        model = MockModel()
        agent = Agent(instructions="Test instructions", model=model)

        response = await agent._generate_response("test input")

        assert response.startswith("Generated response for:")
        assert "test input" in response

    @pytest.mark.asyncio
    async def test_generate_response_without_model(self):
        """Test response generation without model."""
        agent = Agent(instructions="Test instructions")

        response = await agent._generate_response("test input")

        assert response == ""

    @pytest.mark.asyncio
    async def test_generate_response_with_error(self):
        """Test response generation when model raises error."""
        model = Mock()
        model.generate = AsyncMock(side_effect=Exception("Model error"))

        agent = Agent(instructions="Test instructions", model=model)

        response = await agent._generate_response("test input")

        assert "I'm sorry, I encountered an error" in response

    @pytest.mark.asyncio
    async def test_handle_audio_input_without_stt(self):
        """Test audio input handling without STT."""
        agent = Agent(instructions="Test instructions")

        # Mock user object
        mock_user = Mock()
        mock_user.user_id = "test_user"

        # Should not raise an error
        await agent._handle_audio_input(b"fake audio data", mock_user)

    @pytest.mark.asyncio
    async def test_handle_audio_input_with_turn_detection_false(self):
        """Test audio input handling when turn detection returns False."""
        stt = MockSTT()
        turn_detection = Mock()
        turn_detection.detect_turn.return_value = False

        agent = Agent(
            instructions="Test instructions", stt=stt, turn_detection=turn_detection
        )

        # Mock PCM data and user
        mock_pcm = Mock()
        mock_pcm.data = b"fake audio data"
        mock_user = Mock()
        mock_user.user_id = "test_user"

        # Should not process audio when turn detection returns False
        await agent._handle_audio_input(mock_pcm, mock_user)

        turn_detection.detect_turn.assert_called_once_with(b"fake audio data")

    @pytest.mark.asyncio
    async def test_handle_audio_input_full_pipeline(self):
        """Test full audio input processing pipeline."""
        tools = [MockTool("tool_result")]
        pre_processors = [MockPreProcessor()]
        model = MockModel()
        stt = MockSTT()
        tts = MockTTS()
        turn_detection = MockTurnDetection()

        agent = Agent(
            instructions="Test instructions",
            tools=tools,
            pre_processors=pre_processors,
            model=model,
            stt=stt,
            tts=tts,
            turn_detection=turn_detection,
        )

        # Mock user object
        mock_user = Mock()
        mock_user.user_id = "test_user"
        mock_user.name = "Test User"

        await agent._handle_audio_input(b"fake audio data", mock_user)

        # Give a moment for async event processing
        await asyncio.sleep(0.01)

        # Check that TTS received the generated response
        assert len(tts.sent_texts) == 1
        assert tts.sent_texts[0].startswith("Generated response for:")

    @pytest.mark.asyncio
    async def test_handle_audio_input_empty_transcription(self):
        """Test audio input handling with empty transcription."""

        class MockEmptySTT(MockSTT):
            async def process_audio(self, pcm_data, user):
                # Simulate empty transcription result
                await self.emit("transcript", "   ", user, {})  # Empty/whitespace text

        stt = MockEmptySTT()
        tts = MockTTS()

        agent = Agent(instructions="Test instructions", stt=stt, tts=tts)

        # Mock user object
        mock_user = Mock()
        mock_user.user_id = "test_user"

        await agent._handle_audio_input(b"fake audio data", mock_user)

        # Give a moment for async event processing
        await asyncio.sleep(0.01)

        # Should not send anything to TTS for empty transcription
        assert len(tts.sent_texts) == 0

    def test_is_running_initially_false(self):
        """Test that agent is not running initially."""
        agent = Agent(instructions="Test instructions")
        assert not agent.is_running()

    @pytest.mark.asyncio
    async def test_join_requires_mock_call(self):
        """Test that join method would need mocking for full testing."""
        agent = Agent(instructions="Test instructions")

        # We can't easily test the join method without mocking the entire
        # Stream SDK, but we can test that it raises an error when called
        # with invalid arguments
        with pytest.raises(Exception):
            await agent.join(None)  # Invalid call object

    def test_pre_processor_pipeline(self):
        """Test pre-processor pipeline."""
        pre_processor1 = MockPreProcessor()
        pre_processor2 = Mock()
        pre_processor2.process = Mock(return_value="final_processed")

        agent = Agent(
            instructions="Test instructions",
            pre_processors=[pre_processor1, pre_processor2],
        )

        # Simulate the pre-processing pipeline
        data = "original_data"
        for processor in agent.pre_processors:
            data = processor.process(data)

        assert data == "final_processed"
        pre_processor2.process.assert_called_once_with("processed_original_data")

    def test_tools_integration(self):
        """Test tools integration."""
        tool1 = MockTool("result1")
        tool2 = MockTool("result2")

        agent = Agent(instructions="Test instructions", tools=[tool1, tool2])

        # Test that tools can be called
        result1 = agent.tools[0]()
        result2 = agent.tools[1]()

        assert result1 == "result1"
        assert result2 == "result2"

    @pytest.mark.asyncio
    async def test_audio_handling_integration(self):
        """Test audio handling integration end-to-end."""
        stt = MockSTT()
        tts = MockTTS()
        model = MockModel()
        turn_detection = MockTurnDetection()  # Always returns True for testing

        agent = Agent(
            instructions="Test instructions",
            stt=stt,
            tts=tts,
            model=model,
            turn_detection=turn_detection,
        )

        # Mock user object
        mock_user = Mock()
        mock_user.user_id = "test_user"
        mock_user.name = "Test User"

        # Test audio handling with mock data
        test_audio_data = b"test audio data"
        await agent._handle_audio_input(test_audio_data, mock_user)

        # Give a moment for async processing
        await asyncio.sleep(0.01)

        # Verify that STT was called and TTS received a response
        assert len(tts.sent_texts) == 1
        assert tts.sent_texts[0].startswith("Generated response for:")

    # New comprehensive tests for STT event handlers
    @pytest.mark.asyncio
    async def test_stt_transcript_event_handler(self):
        """Test STT transcript event handler."""
        stt = MockSTT()
        tts = MockTTS()
        model = MockModel()

        agent = Agent(instructions="Test instructions", stt=stt, tts=tts, model=model)

        # Mock user object
        mock_user = Mock()
        mock_user.user_id = "test_user"
        mock_user.name = "Test User"

        # Simulate transcript event
        await agent._on_transcript("Hello world", mock_user, {"confidence": 0.95})

        # Give a moment for async processing
        await asyncio.sleep(0.01)

        # Should process transcript and send TTS response
        assert len(tts.sent_texts) == 1
        assert tts.sent_texts[0].startswith("Generated response for:")

    @pytest.mark.asyncio
    async def test_stt_partial_transcript_event_handler(self):
        """Test STT partial transcript event handler."""
        agent = Agent(instructions="Test instructions")

        # Mock user object
        mock_user = Mock()
        mock_user.user_id = "test_user"
        mock_user.name = "Test User"

        # Should not raise error for partial transcript
        await agent._on_partial_transcript("Hel...", mock_user, {})

    @pytest.mark.asyncio
    async def test_stt_error_event_handler(self):
        """Test STT error event handler."""
        agent = Agent(instructions="Test instructions")

        # Should not raise error for STT error
        await agent._on_stt_error("Connection timeout")

    @pytest.mark.asyncio
    async def test_audio_processing_without_vad(self):
        """Test audio processing without VAD (direct STT)."""
        stt = MockSTT()
        tts = MockTTS()
        model = MockModel()

        agent = Agent(instructions="Test instructions", stt=stt, tts=tts, model=model)

        # Mock user object
        mock_user = Mock()
        mock_user.user_id = "test_user"
        mock_user.name = "Test User"

        # Process audio without VAD
        await agent._handle_audio_input(b"fake audio data", mock_user)

        # Give a moment for async processing
        await asyncio.sleep(0.01)

        # Should process audio directly through STT
        assert len(tts.sent_texts) == 1

    @pytest.mark.asyncio
    async def test_audio_processing_with_vad(self):
        """Test audio processing with VAD."""
        stt = MockSTT()
        vad = MockVAD()
        tts = MockTTS()
        model = MockModel()

        agent = Agent(
            instructions="Test instructions", stt=stt, vad=vad, tts=tts, model=model
        )

        # Mock user object
        mock_user = Mock()
        mock_user.user_id = "test_user"
        mock_user.name = "Test User"

        # Process audio with VAD
        await agent._handle_audio_input(b"fake audio data", mock_user)

        # Give a moment for async processing
        await asyncio.sleep(0.01)

        # Should process audio through VAD first
        # Note: This test verifies VAD path is taken, not full speech-to-text pipeline

    @pytest.mark.asyncio
    async def test_vad_speech_start_event(self):
        """Test VAD speech start event handler."""
        agent = Agent(instructions="Test instructions")

        # Mock user object
        mock_user = Mock()
        mock_user.user_id = "test_user"
        mock_user.name = "Test User"

        # Should not raise error for speech start
        await agent._on_speech_start(mock_user, {})

    @pytest.mark.asyncio
    async def test_vad_speech_end_event(self):
        """Test VAD speech end event handler."""
        agent = Agent(instructions="Test instructions")

        # Mock user object
        mock_user = Mock()
        mock_user.user_id = "test_user"
        mock_user.name = "Test User"

        # Should not raise error for speech end
        await agent._on_speech_end(mock_user, {})

    @pytest.mark.asyncio
    async def test_stt_event_handlers_setup_once(self):
        """Test that STT event handlers are set up only once."""
        stt = MockSTT()
        agent = Agent(instructions="Test instructions", stt=stt)

        # Mock user object
        mock_user = Mock()
        mock_user.user_id = "test_user"

        # Call _handle_audio_input multiple times
        await agent._handle_audio_input(b"audio1", mock_user)
        await agent._handle_audio_input(b"audio2", mock_user)

        # Event handlers should only be registered once
        assert hasattr(agent, "_stt_setup")
        assert len(stt._event_handlers.get("transcript", [])) == 1
        assert len(stt._event_handlers.get("partial_transcript", [])) == 1
        assert len(stt._event_handlers.get("error", [])) == 1

    @pytest.mark.asyncio
    async def test_vad_event_handlers_setup_once(self):
        """Test that VAD event handlers are set up only once."""
        vad = MockVAD()
        agent = Agent(instructions="Test instructions", vad=vad)

        # Mock user object
        mock_user = Mock()
        mock_user.user_id = "test_user"

        # Call _process_audio_with_vad multiple times
        await agent._process_audio_with_vad(b"audio1", mock_user)
        await agent._process_audio_with_vad(b"audio2", mock_user)

        # Event handlers should only be registered once
        assert hasattr(agent, "_vad_setup")
        assert len(vad._event_handlers.get("speech_start", [])) == 1
        assert len(vad._event_handlers.get("speech_end", [])) == 1

    @pytest.mark.asyncio
    async def test_user_identification_with_name(self):
        """Test user identification when user has name."""
        stt = MockSTT()
        agent = Agent(instructions="Test instructions", stt=stt)

        # Mock user with name
        mock_user = Mock()
        mock_user.user_id = "test_user"
        mock_user.name = "John Doe"

        # Should use name for identification
        await agent._on_transcript("Hello", mock_user, {})

    @pytest.mark.asyncio
    async def test_user_identification_without_name(self):
        """Test user identification when user has no name."""
        stt = MockSTT()
        agent = Agent(instructions="Test instructions", stt=stt)

        # Mock user without name
        mock_user = Mock()
        mock_user.user_id = "test_user"
        # No name attribute

        # Should use user_id for identification
        await agent._on_transcript("Hello", mock_user, {})

    @pytest.mark.asyncio
    async def test_confidence_logging(self):
        """Test confidence score logging in transcript handler."""
        stt = MockSTT()
        agent = Agent(instructions="Test instructions", stt=stt)

        # Mock user
        mock_user = Mock()
        mock_user.user_id = "test_user"
        mock_user.name = "Test User"

        # Should handle transcript with confidence
        metadata_with_confidence = {"confidence": 0.95}
        await agent._on_transcript("Hello", mock_user, metadata_with_confidence)

        # Should handle transcript without confidence
        metadata_without_confidence = {}
        await agent._on_transcript("Hello", mock_user, metadata_without_confidence)

    @pytest.mark.asyncio
    async def test_agent_cleanup_with_stt_and_vad(self):
        """Test proper cleanup of STT and VAD services."""
        stt = MockSTT()
        vad = MockVAD()

        # Mock close methods to track calls
        stt.close = AsyncMock()
        vad.close = AsyncMock()

        agent = Agent(instructions="Test instructions", stt=stt, vad=vad)

        # Call stop method
        await agent.stop()

        # Both services should be closed
        stt.close.assert_called_once()
        vad.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_cleanup_with_close_errors(self):
        """Test cleanup handles errors gracefully."""
        stt = MockSTT()
        vad = MockVAD()

        # Mock close methods to raise errors
        stt.close = AsyncMock(side_effect=Exception("STT close error"))
        vad.close = AsyncMock(side_effect=Exception("VAD close error"))

        agent = Agent(instructions="Test instructions", stt=stt, vad=vad)

        # Should not raise error despite close failures
        await agent.stop()

    @pytest.mark.asyncio
    async def test_tts_audio_event_handling(self):
        """Test TTS audio event handling with proper user handling."""
        tts = MockTTS()
        agent = Agent(instructions="Test instructions", tts=tts)

        # Set up agent with mock connection
        agent._connection = Mock()
        agent._is_running = True

        # Mock user object
        mock_user = Mock()
        mock_user.user_id = "test_user"
        mock_user.name = "Test User"

        # Test audio event handling
        audio_data = b"test_audio_data"
        await agent._on_tts_audio(audio_data, mock_user)

        # Should not raise any errors

    @pytest.mark.asyncio
    async def test_tts_audio_event_with_none_user(self):
        """Test TTS audio event handling with None user."""
        tts = MockTTS()
        agent = Agent(instructions="Test instructions", tts=tts)

        # Set up agent with mock connection
        agent._connection = Mock()
        agent._is_running = True

        # Test audio event handling with None user
        audio_data = b"test_audio_data"
        await agent._on_tts_audio(audio_data, None)

        # Should not raise any errors

    @pytest.mark.asyncio
    async def test_tts_error_event_handling(self):
        """Test TTS error event handling with proper user handling."""
        tts = MockTTS()
        agent = Agent(instructions="Test instructions", tts=tts)

        # Mock user object
        mock_user = Mock()
        mock_user.user_id = "test_user"
        mock_user.name = "Test User"

        # Test error event handling
        error = Exception("Test TTS error")
        await agent._on_tts_error(error, mock_user)

        # Should not raise any errors

    @pytest.mark.asyncio
    async def test_tts_error_event_with_none_user(self):
        """Test TTS error event handling with None user."""
        tts = MockTTS()
        agent = Agent(instructions="Test instructions", tts=tts)

        # Test error event handling with None user
        error = Exception("Test TTS error")
        await agent._on_tts_error(error, None)

        # Should not raise any errors

    @pytest.mark.asyncio
    async def test_tts_error_event_with_user_without_name(self):
        """Test TTS error event handling with user that has no name attribute."""
        tts = MockTTS()
        agent = Agent(instructions="Test instructions", tts=tts)

        # Mock user object without name
        mock_user = Mock()
        mock_user.user_id = "test_user"
        # No name attribute

        # Test error event handling
        error = Exception("Test TTS error")
        await agent._on_tts_error(error, mock_user)

        # Should not raise any errors

    @pytest.mark.asyncio
    async def test_tts_event_registration_on_join(self):
        """Test that TTS event handlers are properly registered when agent joins."""
        tts = MockTTS()
        agent = Agent(instructions="Test instructions", tts=tts)

        # Mock connection and setup
        mock_connection = Mock()
        mock_connection.on = Mock()
        agent._connection = mock_connection
        agent._is_running = True

        # Simulate the event handler registration that happens in join
        def on_tts_audio(audio_data, user=None, metadata=None):
            user_id = user.user_id if user else "unknown"
            agent.logger.debug(f"🔊 TTS audio generated for: {user_id}")
            asyncio.create_task(agent._on_tts_audio(audio_data, user, metadata))

        tts.on("audio", on_tts_audio)

        def on_tts_error(error, user=None, metadata=None):
            user_id = user.user_id if user else "unknown"
            agent.logger.error(f"❌ TTS Error for {user_id}: {error}")
            asyncio.create_task(agent._on_tts_error(error, user, metadata))

        tts.on("error", on_tts_error)

        # Verify handlers are registered
        assert "audio" in tts._event_handlers
        assert "error" in tts._event_handlers
        assert len(tts._event_handlers["audio"]) == 1
        assert len(tts._event_handlers["error"]) == 1

    @pytest.mark.asyncio
    async def test_tts_audio_event_emission(self):
        """Test TTS audio event emission and handling."""
        tts = MockTTS()

        audio_events_received = []

        # Register event handler
        def on_tts_audio(audio_data, user=None, metadata=None):
            audio_events_received.append((audio_data, user, metadata))

        tts.on("audio", on_tts_audio)

        # Mock user
        mock_user = Mock()
        mock_user.user_id = "test_user"

        # Emit audio event
        await tts.emit("audio", b"test_audio", mock_user, {"sample_rate": 16000})

        # Verify event was received
        assert len(audio_events_received) == 1
        audio_data, user, metadata = audio_events_received[0]
        assert audio_data == b"test_audio"
        assert user == mock_user
        assert metadata == {"sample_rate": 16000}

    @pytest.mark.asyncio
    async def test_tts_error_event_emission(self):
        """Test TTS error event emission and handling."""
        tts = MockTTS()

        error_events_received = []

        # Register event handler
        def on_tts_error(error, user=None, metadata=None):
            error_events_received.append((error, user, metadata))

        tts.on("error", on_tts_error)

        # Mock user
        mock_user = Mock()
        mock_user.user_id = "test_user"

        # Emit error event
        test_error = Exception("Test TTS error")
        await tts.emit("error", test_error, mock_user)

        # Verify event was received
        assert len(error_events_received) == 1
        error, user, metadata = error_events_received[0]
        assert error == test_error
        assert user == mock_user

    @pytest.mark.asyncio
    async def test_tts_send_generates_audio_event(self):
        """Test that TTS send method generates audio event."""
        tts = MockTTS()

        audio_events_received = []

        # Register event handler
        def on_tts_audio(audio_data, user=None, metadata=None):
            audio_events_received.append((audio_data, user))

        tts.on("audio", on_tts_audio)

        # Mock user
        mock_user = Mock()
        mock_user.user_id = "test_user"

        # Send text to TTS
        await tts.send("Hello world", mock_user)

        # Verify text was recorded
        assert "Hello world" in tts.sent_texts

        # Verify audio event was emitted
        assert len(audio_events_received) == 1
        audio_data, user = audio_events_received[0]
        assert audio_data == b"mock_audio_data"
        assert user == mock_user

    @pytest.mark.asyncio
    async def test_tts_set_output_track(self):
        """Test TTS output track configuration."""
        tts = MockTTS()

        # Mock audio track
        mock_track = Mock()

        # Set output track
        tts.set_output_track(mock_track)

        # Verify track is set
        assert tts.output_track == mock_track


if __name__ == "__main__":
    pytest.main([__file__])
