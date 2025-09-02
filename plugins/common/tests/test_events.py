import pytest
import json
from datetime import datetime

from getstream.plugins.common.events import (
    # Base events
    BaseEvent,
    EventType,
    ConnectionState,
    AudioFormat,
    create_event,
    
    # STT Events
    STTTranscriptEvent,
    STTPartialTranscriptEvent,
    STTErrorEvent,
    STTConnectionEvent,
    
    # TTS Events
    TTSAudioEvent,
    TTSSynthesisStartEvent,
    TTSSynthesisCompleteEvent,
    TTSErrorEvent,
    TTSConnectionEvent,
    
    # STS Events
    STSConnectedEvent,
    STSDisconnectedEvent,
    STSAudioInputEvent,
    STSAudioOutputEvent,
    STSTranscriptEvent,
    STSResponseEvent,
    STSConversationItemEvent,
    STSErrorEvent,
    
    # VAD Events
    VADSpeechStartEvent,
    VADSpeechEndEvent,
    VADAudioEvent,
    VADPartialEvent,
    VADErrorEvent,
    
    # Generic Events
    PluginInitializedEvent,
    PluginClosedEvent,
    PluginErrorEvent,
)

from getstream.plugins.common.event_serialization import (
    serialize_event,
    serialize_events,
    deserialize_event,
)


class TestBaseEvent:
    """Test the base event functionality."""
    
    def test_base_event_creation(self):
        """Test that base events can be created with minimal parameters."""
        event = BaseEvent(event_type=EventType.STT_TRANSCRIPT)
        assert event.event_id is not None
        assert event.timestamp is not None
        assert isinstance(event.timestamp, datetime)
        assert event.session_id is None
        assert event.user_metadata is None
        assert event.plugin_name is None
        assert event.plugin_version is None
    
    def test_base_event_with_optional_params(self):
        """Test base event creation with optional parameters."""
        session_id = "test-session-123"
        user_metadata = {"user_id": "user123"}
        plugin_name = "test_plugin"
        plugin_version = "1.0.0"
        
        event = BaseEvent(
            event_type=EventType.STT_TRANSCRIPT,
            session_id=session_id,
            user_metadata=user_metadata,
            plugin_name=plugin_name,
            plugin_version=plugin_version
        )
        
        assert event.session_id == session_id
        assert event.user_metadata == user_metadata
        assert event.plugin_name == plugin_name
        assert event.plugin_version == plugin_version


class TestSTTEvents:
    """Test STT-related events."""
    
    def test_stt_transcript_event_creation(self):
        """Test STT transcript event creation."""
        event = STTTranscriptEvent(text="Hello world")
        assert event.event_type == EventType.STT_TRANSCRIPT
        assert event.text == "Hello world"
        assert event.confidence is None  # Should be None by default
        assert event.language is None
        assert event.processing_time_ms is None
        assert event.audio_duration_ms is None
        assert event.model_name is None
        assert event.words is None
        assert event.is_final is True
    
    def test_stt_transcript_event_with_confidence(self):
        """Test STT transcript event with confidence."""
        event = STTTranscriptEvent(
            text="Hello world",
            confidence=0.95,
            language="en",
            processing_time_ms=100.0,
            audio_duration_ms=2000.0,
            model_name="test-model"
        )
        assert event.confidence == 0.95
        assert event.language == "en"
        assert event.processing_time_ms == 100.0
        assert event.audio_duration_ms == 2000.0
        assert event.model_name == "test-model"
    
    def test_stt_transcript_event_validation(self):
        """Test STT transcript event validation."""
        # Should raise ValueError for empty text
        with pytest.raises(ValueError, match="Transcript text cannot be empty"):
            STTTranscriptEvent(text="")
    
        # Whitespace-only text should pass validation (current behavior)
        event = STTTranscriptEvent(text="   ")
        assert event.text == "   "
        
        # Test that non-empty text works
        event = STTTranscriptEvent(text="Valid text")
        assert event.text == "Valid text"
    
    def test_stt_partial_transcript_event(self):
        """Test STT partial transcript event."""
        event = STTPartialTranscriptEvent(text="Hello")
        assert event.event_type == EventType.STT_PARTIAL_TRANSCRIPT
        assert event.text == "Hello"
        assert event.confidence is None
        assert event.is_final is False
    
    def test_stt_error_event(self):
        """Test STT error event."""
        error = Exception("Test error")
        event = STTErrorEvent(
            error=error,
            error_code="TEST_001",
            context="test context",
            retry_count=3,
            is_recoverable=False
        )
        assert event.event_type == EventType.STT_ERROR
        assert event.error == error
        assert event.error_code == "TEST_001"
        assert event.context == "test context"
        assert event.retry_count == 3
        assert event.is_recoverable is False
        assert event.error_message == "Test error"
    
    def test_stt_connection_event(self):
        """Test STT connection event."""
        event = STTConnectionEvent(
            connection_state=ConnectionState.CONNECTED,
            provider="test_provider",
            details={"test": "detail"},
            reconnect_attempts=2
        )
        assert event.event_type == EventType.STT_CONNECTION
        assert event.connection_state == ConnectionState.CONNECTED
        assert event.provider == "test_provider"
        assert event.details == {"test": "detail"}
        assert event.reconnect_attempts == 2


class TestTTSEvents:
    """Test TTS-related events."""
    
    def test_tts_audio_event_creation(self):
        """Test TTS audio event creation."""
        audio_data = b"test audio data"
        event = TTSAudioEvent(
            audio_data=audio_data,
            audio_format=AudioFormat.PCM_S16,
            sample_rate=22050,
            channels=2,
            chunk_index=5,
            is_final_chunk=False
        )
        assert event.event_type == EventType.TTS_AUDIO
        assert event.audio_data == audio_data
        assert event.audio_format == AudioFormat.PCM_S16
        assert event.sample_rate == 22050
        assert event.channels == 2
        assert event.chunk_index == 5
        assert event.is_final_chunk is False
    
    def test_tts_synthesis_start_event(self):
        """Test TTS synthesis start event."""
        event = TTSSynthesisStartEvent(
            text="Hello world",
            model_name="test-model",
            voice_id="voice123",
            estimated_duration_ms=5000.0
        )
        assert event.event_type == EventType.TTS_SYNTHESIS_START
        assert event.text == "Hello world"
        assert event.model_name == "test-model"
        assert event.voice_id == "voice123"
        assert event.estimated_duration_ms == 5000.0
        assert event.synthesis_id is not None
    
    def test_tts_synthesis_complete_event(self):
        """Test TTS synthesis complete event."""
        event = TTSSynthesisCompleteEvent(
            synthesis_id="synth123",
            text="Hello world",
            total_audio_bytes=1024,
            synthesis_time_ms=2500.0,
            audio_duration_ms=5000.0,
            chunk_count=10,
            real_time_factor=0.5
        )
        assert event.event_type == EventType.TTS_SYNTHESIS_COMPLETE
        assert event.synthesis_id == "synth123"
        assert event.text == "Hello world"
        assert event.total_audio_bytes == 1024
        assert event.synthesis_time_ms == 2500.0
        assert event.audio_duration_ms == 5000.0
        assert event.chunk_count == 10
        assert event.real_time_factor == 0.5
    
    def test_tts_error_event(self):
        """Test TTS error event."""
        error = Exception("TTS synthesis failed")
        event = TTSErrorEvent(
            error=error,
            error_code="TTS_001",
            context="synthesis",
            is_recoverable=True
        )
        assert event.event_type == EventType.TTS_ERROR
        assert event.error == error
        assert event.error_code == "TTS_001"
        assert event.context == "synthesis"
        assert event.is_recoverable is True
    
    def test_tts_connection_event(self):
        """Test TTS connection event."""
        event = TTSConnectionEvent(
            connection_state=ConnectionState.DISCONNECTED,
            provider="tts_provider",
            details={"reason": "timeout"}
        )
        assert event.event_type == EventType.TTS_CONNECTION
        assert event.connection_state == ConnectionState.DISCONNECTED
        assert event.provider == "tts_provider"
        assert event.details == {"reason": "timeout"}


class TestSTSEvents:
    """Test STS-related events."""
    
    def test_sts_connected_event(self):
        """Test STS connected event."""
        event = STSConnectedEvent(
            provider="sts_provider",
            session_config={"endpoint": "wss://test.com"}
        )
        assert event.event_type == EventType.STS_CONNECTED
        assert event.provider == "sts_provider"
        assert event.session_config == {"endpoint": "wss://test.com"}
    
    def test_sts_disconnected_event(self):
        """Test STS disconnected event."""
        event = STSDisconnectedEvent(
            provider="sts_provider",
            reason="user_disconnect",
            was_clean=True
        )
        assert event.event_type == EventType.STS_DISCONNECTED
        assert event.provider == "sts_provider"
        assert event.reason == "user_disconnect"
        assert event.was_clean is True
    
    def test_sts_audio_input_event(self):
        """Test STS audio input event."""
        event = STSAudioInputEvent(
            audio_data=b"input audio",
            sample_rate=16000,
            channels=1
        )
        assert event.event_type == EventType.STS_AUDIO_INPUT
        assert event.audio_data == b"input audio"
        assert event.sample_rate == 16000
        assert event.channels == 1
    
    def test_sts_audio_output_event(self):
        """Test STS audio output event."""
        event = STSAudioOutputEvent(
            audio_data=b"output audio",
            sample_rate=16000,
            channels=1
        )
        assert event.event_type == EventType.STS_AUDIO_OUTPUT
        assert event.audio_data == b"output audio"
        assert event.sample_rate == 16000
        assert event.channels == 1
    
    def test_sts_transcript_event(self):
        """Test STS transcript event."""
        event = STSTranscriptEvent(
            text="Hello world",
            confidence=0.95,
            is_user=True
        )
        assert event.event_type == EventType.STS_TRANSCRIPT
        assert event.text == "Hello world"
        assert event.confidence == 0.95
        assert event.is_user is True
    
    def test_sts_response_event(self):
        """Test STS response event."""
        event = STSResponseEvent(
            text="Response text",
            is_complete=True
        )
        assert event.event_type == EventType.STS_RESPONSE
        assert event.text == "Response text"
        assert event.is_complete is True
        assert event.response_id is not None
    
    def test_sts_conversation_item_event(self):
        """Test STS conversation item event."""
        event = STSConversationItemEvent(
            item_type="message",
            status="completed",
            role="user",
            content=[{"type": "text", "text": "User message"}]
        )
        assert event.event_type == EventType.STS_CONVERSATION_ITEM
        assert event.item_type == "message"
        assert event.status == "completed"
        assert event.role == "user"
        assert event.content == [{"type": "text", "text": "User message"}]
    
    def test_sts_error_event(self):
        """Test STS error event."""
        error = Exception("STS error")
        event = STSErrorEvent(
            error=error,
            error_code="STS_001",
            context="conversation"
        )
        assert event.event_type == EventType.STS_ERROR
        assert event.error == error
        assert event.error_code == "STS_001"
        assert event.context == "conversation"


class TestVADEvents:
    """Test VAD-related events."""
    
    def test_vad_speech_start_event(self):
        """Test VAD speech start event."""
        event = VADSpeechStartEvent(
            speech_probability=0.8,
            activation_threshold=0.5,
            frame_count=10
        )
        assert event.event_type == EventType.VAD_SPEECH_START
        assert event.speech_probability == 0.8
        assert event.activation_threshold == 0.5
        assert event.frame_count == 10
    
    def test_vad_speech_end_event(self):
        """Test VAD speech end event."""
        event = VADSpeechEndEvent(
            speech_probability=0.2,
            deactivation_threshold=0.3,
            total_speech_duration_ms=2500.0,
            total_frames=100
        )
        assert event.event_type == EventType.VAD_SPEECH_END
        assert event.speech_probability == 0.2
        assert event.deactivation_threshold == 0.3
        assert event.total_speech_duration_ms == 2500.0
        assert event.total_frames == 100
    
    def test_vad_audio_event(self):
        """Test VAD audio event."""
        event = VADAudioEvent(
            audio_data=b"speech audio",
            sample_rate=16000,
            audio_format="s16",
            channels=1,
            duration_ms=1000.0,
            speech_probability=0.9,
            frame_count=50
        )
        assert event.event_type == EventType.VAD_AUDIO
        assert event.audio_data == b"speech audio"
        assert event.sample_rate == 16000
        assert event.audio_format == "s16"
        assert event.channels == 1
        assert event.duration_ms == 1000.0
        assert event.speech_probability == 0.9
        assert event.frame_count == 50
    
    def test_vad_partial_event(self):
        """Test VAD partial event."""
        event = VADPartialEvent(
            speech_probability=0.7,
            frame_count=25,
            is_speech_active=True
        )
        assert event.event_type == EventType.VAD_PARTIAL
        assert event.speech_probability == 0.7
        assert event.frame_count == 25
        assert event.is_speech_active is True
    
    def test_vad_error_event(self):
        """Test VAD error event."""
        error = Exception("VAD processing error")
        event = VADErrorEvent(
            error=error,
            error_code="VAD_001",
            context="audio_processing",
            frame_data_available=False
        )
        assert event.event_type == EventType.VAD_ERROR
        assert event.error == error
        assert event.error_code == "VAD_001"
        assert event.context == "audio_processing"
        assert event.frame_data_available is False


class TestGenericEvents:
    """Test generic plugin events."""
    
    def test_plugin_initialized_event(self):
        """Test plugin initialized event."""
        event = PluginInitializedEvent(
            plugin_type="STT",
            provider="test_provider",
            configuration={"setting": "value"},
            capabilities=["transcription", "language_detection"]
        )
        assert event.event_type == EventType.PLUGIN_INITIALIZED
        assert event.plugin_type == "STT"
        assert event.provider == "test_provider"
        assert event.configuration == {"setting": "value"}
        assert event.capabilities == ["transcription", "language_detection"]
    
    def test_plugin_closed_event(self):
        """Test plugin closed event."""
        event = PluginClosedEvent(
            plugin_type="STT",
            provider="test_provider",
            reason="shutdown",
            cleanup_successful=True
        )
        assert event.event_type == EventType.PLUGIN_CLOSED
        assert event.plugin_type == "STT"
        assert event.provider == "test_provider"
        assert event.reason == "shutdown"
        assert event.cleanup_successful is True
    
    def test_plugin_error_event(self):
        """Test plugin error event."""
        error = Exception("Plugin error")
        event = PluginErrorEvent(
            plugin_type="STT",
            provider="test_provider",
            error=error,
            error_code="PLUGIN_001",
            context="initialization",
            is_fatal=False
        )
        assert event.event_type == EventType.PLUGIN_ERROR
        assert event.plugin_type == "STT"
        assert event.provider == "test_provider"
        assert event.error == error
        assert event.error_code == "PLUGIN_001"
        assert event.context == "initialization"
        assert event.is_fatal is False


class TestEventEnums:
    """Test event enums and constants."""
    
    def test_event_types(self):
        """Test that all event types are defined."""
        expected_types = [
            "stt_transcript",
            "stt_partial_transcript", 
            "stt_error",
            "stt_connection",
            "tts_audio",
            "tts_synthesis_start",
            "tts_synthesis_complete",
            "tts_error",
            "tts_connection",
            "sts_connected",
            "sts_disconnected",
            "sts_audio_input",
            "sts_audio_output",
            "sts_transcript",
            "sts_response",
            "sts_conversation_item",
            "sts_error",
            "vad_speech_start",
            "vad_speech_end",
            "vad_audio",
            "vad_partial",
            "vad_error",
            "plugin_initialized",
            "plugin_closed",
            "plugin_error"
        ]
        
        for expected_type in expected_types:
            assert hasattr(EventType, expected_type.upper().replace('-', '_')), f"Missing event type: {expected_type}"
    
    def test_connection_states(self):
        """Test connection state enum values."""
        assert ConnectionState.CONNECTING.value == "connecting"
        assert ConnectionState.CONNECTED.value == "connected"
        assert ConnectionState.RECONNECTING.value == "reconnecting"
        assert ConnectionState.ERROR.value == "error"
    
    def test_audio_formats(self):
        """Test audio format enum values."""
        assert AudioFormat.PCM_S16.value == "s16"
        assert AudioFormat.PCM_F32.value == "f32"
        assert AudioFormat.WAV.value == "wav"
        assert AudioFormat.MP3.value == "mp3"
        assert AudioFormat.OGG.value == "ogg"


class TestEventEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_event_with_none_values(self):
        """Test that events can handle None values gracefully."""
        event = STTTranscriptEvent(
            text="test",
            confidence=None,
            language=None,
            processing_time_ms=None
        )
        assert event.confidence is None
        assert event.language is None
        assert event.processing_time_ms is None
    
    def test_event_with_empty_strings(self):
        """Test that events can handle empty strings."""
        event = STTTranscriptEvent(
            text="test",
            language="",
            model_name=""
        )
        assert event.language == ""
        assert event.model_name == ""
    
    def test_event_with_zero_values(self):
        """Test that events can handle zero values."""
        event = STTTranscriptEvent(
            text="test",
            confidence=0.0,
            processing_time_ms=0.0,
            audio_duration_ms=0.0
        )
        assert event.confidence == 0.0
        assert event.processing_time_ms == 0.0
        assert event.audio_duration_ms == 0.0
    
    def test_event_serialization_compatibility(self):
        """Test that events can be converted to dict for serialization."""
        event = STTTranscriptEvent(
            text="test",
            confidence=0.95,
            language="en"
        )
        
        # Test that we can access all fields
        event_dict = {
            'text': event.text,
            'confidence': event.confidence,
            'language': event.language,
            'event_type': event.event_type.value,
            'event_id': event.event_id,
            'timestamp': event.timestamp.isoformat(),
            'is_final': event.is_final
        }
        
        assert event_dict['text'] == "test"
        assert event_dict['confidence'] == 0.95
        assert event_dict['language'] == "en"
        assert event_dict['event_type'] == "stt_transcript"
        assert event_dict['is_final'] is True


class TestEventMetrics:
    """Test event metrics calculation functions."""
    
    def test_calculate_stt_metrics_empty_events(self):
        """Test STT metrics calculation with no events."""
        from getstream.plugins.common.event_metrics import calculate_stt_metrics
        
        metrics = calculate_stt_metrics([])
        assert metrics == {"total_transcripts": 0}
    
    def test_calculate_stt_metrics_basic_counts(self):
        """Test STT metrics calculation with basic event counts."""
        from getstream.plugins.common.event_metrics import calculate_stt_metrics
        
        events = [
            STTTranscriptEvent(text="First transcript", is_final=True),
            STTTranscriptEvent(text="Second transcript", is_final=True),
            STTPartialTranscriptEvent(text="Partial transcript", is_final=False),
            STTErrorEvent(error=Exception("Test error"))  # Should be ignored
        ]
        
        metrics = calculate_stt_metrics(events)
        
        assert metrics["total_transcripts"] == 3
        assert metrics["final_transcripts"] == 2
        assert metrics["partial_transcripts"] == 1
    
    def test_calculate_stt_metrics_with_processing_times(self):
        """Test STT metrics calculation with processing time data."""
        from getstream.plugins.common.event_metrics import calculate_stt_metrics
        
        events = [
            STTTranscriptEvent(text="Fast", processing_time_ms=50.0),
            STTTranscriptEvent(text="Medium", processing_time_ms=100.0),
            STTTranscriptEvent(text="Slow", processing_time_ms=200.0),
            STTPartialTranscriptEvent(text="Partial", processing_time_ms=75.0)
        ]
        
        metrics = calculate_stt_metrics(events)
        
        assert metrics["total_transcripts"] == 4
        assert metrics["avg_processing_time_ms"] == 106.25  # (50+100+200+75)/4
        assert metrics["min_processing_time_ms"] == 50.0
        assert metrics["max_processing_time_ms"] == 200.0
    
    def test_calculate_stt_metrics_with_confidence(self):
        """Test STT metrics calculation with confidence data."""
        from getstream.plugins.common.event_metrics import calculate_stt_metrics
        
        events = [
            STTTranscriptEvent(text="High confidence", confidence=0.95),
            STTTranscriptEvent(text="Medium confidence", confidence=0.75),
            STTTranscriptEvent(text="Low confidence", confidence=0.55),
            STTPartialTranscriptEvent(text="Partial", confidence=0.80)
        ]
        
        metrics = calculate_stt_metrics(events)
        
        assert metrics["total_transcripts"] == 4
        assert metrics["avg_confidence"] == 0.7625  # (0.95+0.75+0.55+0.80)/4
        assert metrics["min_confidence"] == 0.55
        assert metrics["max_confidence"] == 0.95
    
    def test_calculate_stt_metrics_mixed_data(self):
        """Test STT metrics calculation with mixed data (some missing values)."""
        from getstream.plugins.common.event_metrics import calculate_stt_metrics
        
        events = [
            STTTranscriptEvent(text="Complete", confidence=0.9, processing_time_ms=100.0),
            STTTranscriptEvent(text="No confidence", processing_time_ms=150.0),
            STTTranscriptEvent(text="No processing time", confidence=0.8),
            STTPartialTranscriptEvent(text="Partial only")
        ]
        
        metrics = calculate_stt_metrics(events)
        
        assert metrics["total_transcripts"] == 4
        assert metrics["final_transcripts"] == 3
        assert metrics["partial_transcripts"] == 1
        
        # Only events with processing_time_ms should be included
        assert metrics["avg_processing_time_ms"] == 125.0  # (100+150)/2
        assert metrics["min_processing_time_ms"] == 100.0
        assert metrics["max_processing_time_ms"] == 150.0
        
        # Only events with confidence should be included
        assert pytest.approx(metrics["avg_confidence"], 0.001) == 0.85  # (0.9+0.8)/2
        assert metrics["min_confidence"] == 0.8
        assert metrics["max_confidence"] == 0.9
    
    def test_calculate_tts_metrics_empty_events(self):
        """Test TTS metrics calculation with no events."""
        from getstream.plugins.common.event_metrics import calculate_tts_metrics
        
        metrics = calculate_tts_metrics([])
        assert metrics == {
            "total_audio_chunks": 0,
            "total_syntheses": 0,
            "completed_syntheses": 0
        }
    
    def test_calculate_tts_metrics_basic_counts(self):
        """Test TTS metrics calculation with basic event counts."""
        from getstream.plugins.common.event_metrics import calculate_tts_metrics
        
        events = [
            TTSAudioEvent(audio_data=b"chunk1"),
            TTSAudioEvent(audio_data=b"chunk2"),
            TTSSynthesisStartEvent(text="Start synthesis"),
            TTSSynthesisCompleteEvent(synthesis_id="synth1"),
            TTSErrorEvent(error=Exception("TTS error"))  # Should be ignored
        ]
        
        metrics = calculate_tts_metrics(events)
        
        assert metrics["total_audio_chunks"] == 2
        assert metrics["total_syntheses"] == 1
        assert metrics["completed_syntheses"] == 1
    
    def test_calculate_tts_metrics_with_synthesis_times(self):
        """Test TTS metrics calculation with synthesis time data."""
        from getstream.plugins.common.event_metrics import calculate_tts_metrics
        
        events = [
            TTSSynthesisCompleteEvent(synthesis_id="fast", synthesis_time_ms=100.0),
            TTSSynthesisCompleteEvent(synthesis_id="medium", synthesis_time_ms=200.0),
            TTSSynthesisCompleteEvent(synthesis_id="slow", synthesis_time_ms=300.0)
        ]
        
        metrics = calculate_tts_metrics(events)
        
        assert metrics["completed_syntheses"] == 3
        assert metrics["avg_synthesis_time_ms"] == 200.0  # (100+200+300)/3
        assert metrics["min_synthesis_time_ms"] == 100.0
        assert metrics["max_synthesis_time_ms"] == 300.0
    
    def test_calculate_tts_metrics_with_real_time_factors(self):
        """Test TTS metrics calculation with real-time factor data."""
        from getstream.plugins.common.event_metrics import calculate_tts_metrics
        
        events = [
            TTSSynthesisCompleteEvent(synthesis_id="rt1", real_time_factor=0.5),
            TTSSynthesisCompleteEvent(synthesis_id="rt2", real_time_factor=1.0),
            TTSSynthesisCompleteEvent(synthesis_id="rt3", real_time_factor=1.5),
            TTSSynthesisCompleteEvent(synthesis_id="rt4")  # No real_time_factor
        ]
        
        metrics = calculate_tts_metrics(events)
        
        assert metrics["completed_syntheses"] == 4
        assert metrics["avg_real_time_factor"] == 1.0  # (0.5+1.0+1.5)/3 (only 3 have values)
        assert metrics["min_real_time_factor"] == 0.5
        assert metrics["max_real_time_factor"] == 1.5
    
    def test_calculate_vad_metrics_empty_events(self):
        """Test VAD metrics calculation with no events."""
        from getstream.plugins.common.event_metrics import calculate_vad_metrics
        
        metrics = calculate_vad_metrics([])
        assert metrics == {
            "total_speech_segments": 0,
            "total_partial_events": 0
        }
    
    def test_calculate_vad_metrics_basic_counts(self):
        """Test VAD metrics calculation with basic event counts."""
        from getstream.plugins.common.event_metrics import calculate_vad_metrics
        
        events = [
            VADAudioEvent(audio_data=b"speech1"),
            VADAudioEvent(audio_data=b"speech2"),
            VADPartialEvent(speech_probability=0.8),
            VADPartialEvent(speech_probability=0.9),
            VADErrorEvent(error=Exception("VAD error"))  # Should be ignored
        ]
        
        metrics = calculate_vad_metrics(events)
        
        assert metrics["total_speech_segments"] == 2
        assert metrics["total_partial_events"] == 2
    
    def test_calculate_vad_metrics_with_durations_and_probabilities(self):
        """Test VAD metrics calculation with duration and probability data."""
        from getstream.plugins.common.event_metrics import calculate_vad_metrics
        
        events = [
            VADAudioEvent(audio_data=b"short", duration_ms=500.0, speech_probability=0.8),
            VADAudioEvent(audio_data=b"medium", duration_ms=1000.0, speech_probability=0.9),
            VADAudioEvent(audio_data=b"long", duration_ms=2000.0, speech_probability=0.95)
        ]
        
        metrics = calculate_vad_metrics(events)
        
        assert metrics["total_speech_segments"] == 3
        assert pytest.approx(metrics["avg_speech_duration_ms"], 0.01) == 1166.67  # (500+1000+2000)/3
        assert metrics["total_speech_duration_ms"] == 3500.0  # 500+1000+2000
        assert pytest.approx(metrics["avg_speech_probability"], 0.001) == 0.883  # (0.8+0.9+0.95)/3
    
    def test_calculate_vad_metrics_mixed_data(self):
        """Test VAD metrics calculation with mixed data (some missing values)."""
        from getstream.plugins.common.event_metrics import calculate_vad_metrics
        
        events = [
            VADAudioEvent(audio_data=b"complete", duration_ms=1000.0, speech_probability=0.9),
            VADAudioEvent(audio_data=b"no_duration", speech_probability=0.8),
            VADAudioEvent(audio_data=b"no_probability", duration_ms=1500.0),
            VADPartialEvent(speech_probability=0.7)
        ]
        
        metrics = calculate_vad_metrics(events)
        
        assert metrics["total_speech_segments"] == 3
        assert metrics["total_partial_events"] == 1
        
        # Only events with duration_ms should be included in duration calculations
        assert metrics["avg_speech_duration_ms"] == 1250.0  # (1000+1500)/2
        assert metrics["total_speech_duration_ms"] == 2500.0  # 1000+1500
        
        # Only VADAudioEvent events with speech_probability are included (VADPartialEvent is ignored)
        assert pytest.approx(metrics["avg_speech_probability"], 0.001) == 0.85  # (0.9+0.8)/2 = 1.7/2 = 0.85
    
    def test_calculate_metrics_with_realistic_event_sequence(self):
        """Test metrics calculation with a realistic sequence of events."""
        from getstream.plugins.common.event_metrics import (
            calculate_stt_metrics, calculate_tts_metrics, calculate_vad_metrics
        )
        
        # Create a realistic sequence of events
        events = [
            # STT Events
            STTTranscriptEvent(text="Hello", confidence=0.95, processing_time_ms=100.0),
            STTPartialTranscriptEvent(text="Hello wo", confidence=0.85, processing_time_ms=50.0),
            STTTranscriptEvent(text="Hello world", confidence=0.92, processing_time_ms=120.0),
            
            # TTS Events
            TTSSynthesisStartEvent(text="Hello world"),
            TTSAudioEvent(audio_data=b"audio1"),
            TTSAudioEvent(audio_data=b"audio2"),
            TTSSynthesisCompleteEvent(
                synthesis_id="synth1", 
                synthesis_time_ms=200.0, 
                real_time_factor=0.8
            ),
            
            # VAD Events
            VADAudioEvent(audio_data=b"speech1", duration_ms=800.0, speech_probability=0.9),
            VADPartialEvent(speech_probability=0.85),
            VADAudioEvent(audio_data=b"speech2", duration_ms=1200.0, speech_probability=0.95)
        ]
        
        # Calculate all metrics
        stt_metrics = calculate_stt_metrics(events)
        tts_metrics = calculate_tts_metrics(events)
        vad_metrics = calculate_vad_metrics(events)
        
        # Verify STT metrics
        assert stt_metrics["total_transcripts"] == 3
        assert stt_metrics["final_transcripts"] == 2
        assert stt_metrics["partial_transcripts"] == 1
        assert stt_metrics["avg_processing_time_ms"] == 90.0  # (100+50+120)/3
        assert pytest.approx(stt_metrics["avg_confidence"], 0.001) == 0.907  # (0.95+0.85+0.92)/3
        
        # Verify TTS metrics
        assert tts_metrics["total_audio_chunks"] == 2
        assert tts_metrics["total_syntheses"] == 1
        assert tts_metrics["completed_syntheses"] == 1
        assert tts_metrics["avg_synthesis_time_ms"] == 200.0
        assert tts_metrics["avg_real_time_factor"] == 0.8
        
        # Verify VAD metrics
        assert vad_metrics["total_speech_segments"] == 2
        assert vad_metrics["total_partial_events"] == 1
        assert pytest.approx(vad_metrics["avg_speech_duration_ms"], 0.01) == 1000.0  # (800+1200)/2
        assert vad_metrics["total_speech_duration_ms"] == 2000.0  # 800+1200
        assert pytest.approx(vad_metrics["avg_speech_probability"], 0.001) == 0.925  # (0.9+0.95)/2


class TestEventSerialization:
    """Test event serialization and deserialization."""
    
    def test_serialize_stt_transcript_event(self):
        """Test serializing STT transcript event."""
        event = STTTranscriptEvent(
            text="Hello world",
            confidence=0.95,
            language="en",
            processing_time_ms=100.0,
            session_id="test-session",
            user_metadata={"user_id": "123"}
        )
        
        serialized = serialize_event(event)
        
        assert serialized['text'] == "Hello world"
        assert serialized['confidence'] == 0.95
        assert serialized['language'] == "en"
        assert serialized['processing_time_ms'] == 100.0
        assert serialized['session_id'] == "test-session"
        assert serialized['user_metadata'] == {"user_id": "123"}
        assert serialized['event_type'] == "stt_transcript"
        assert 'event_id' in serialized
        assert 'timestamp' in serialized
    
    def test_serialize_event_with_error(self):
        """Test serializing event with error field."""
        error = ValueError("Test error message")
        event = STTErrorEvent(
            error=error,
            error_code="TEST_001",
            context="test context"
        )
        
        serialized = serialize_event(event)
        
        assert serialized['error']['type'] == "ValueError"
        assert serialized['error']['message'] == "Test error message"
        assert serialized['error_code'] == "TEST_001"
        assert serialized['context'] == "test context"
        assert serialized['event_type'] == "stt_error"
    
    def test_serialize_event_with_audio_data(self):
        """Test serializing event with audio data."""
        audio_data = b"fake audio data" * 100  # Create some dummy audio data
        event = TTSAudioEvent(
            audio_data=audio_data,
            sample_rate=22050,
            channels=2
        )
        
        serialized = serialize_event(event)
        
        # Audio data should be replaced with metadata
        assert serialized['audio_data']['size_bytes'] == len(audio_data)
        assert serialized['audio_data']['type'] == 'bytes'
        assert serialized['sample_rate'] == 22050
        assert serialized['channels'] == 2
        assert serialized['event_type'] == "tts_audio"
    
    def test_serialize_events_list(self):
        """Test serializing a list of events to JSON."""
        events = [
            STTTranscriptEvent(text="First transcript"),
            STTPartialTranscriptEvent(text="Partial"),
            STTErrorEvent(error=Exception("Test error"))
        ]
        
        json_string = serialize_events(events)
        
        # Should be valid JSON
        parsed = json.loads(json_string)
        assert len(parsed) == 3
        assert parsed[0]['text'] == "First transcript"
        assert parsed[0]['event_type'] == "stt_transcript"
        assert parsed[1]['text'] == "Partial"
        assert parsed[1]['event_type'] == "stt_partial_transcript"
        assert parsed[2]['event_type'] == "stt_error"
    
    def test_deserialize_stt_transcript_event(self):
        """Test deserializing STT transcript event."""
        original_event = STTTranscriptEvent(
            text="Hello world",
            confidence=0.95,
            language="en",
            processing_time_ms=100.0
        )
        
        # Serialize and then deserialize
        serialized = serialize_event(original_event)
        deserialized = deserialize_event(serialized)
        
        assert isinstance(deserialized, STTTranscriptEvent)
        assert deserialized.text == "Hello world"
        assert deserialized.confidence == 0.95
        assert deserialized.language == "en"
        assert deserialized.processing_time_ms == 100.0
        assert deserialized.event_type == EventType.STT_TRANSCRIPT
    
    def test_deserialize_event_with_error(self):
        """Test deserializing event with error field."""
        original_event = STTErrorEvent(
            error=ValueError("Original error"),
            error_code="TEST_001",
            context="test context"
        )
        
        serialized = serialize_event(original_event)
        deserialized = deserialize_event(serialized)
        
        assert isinstance(deserialized, STTErrorEvent)
        assert isinstance(deserialized.error, Exception)
        assert str(deserialized.error) == "Original error"
        assert deserialized.error_code == "TEST_001"
        assert deserialized.context == "test context"
    
    def test_deserialize_event_with_audio_data_placeholder(self):
        """Test deserializing event with audio data placeholder."""
        original_event = TTSAudioEvent(
            audio_data=b"test audio",
            sample_rate=16000,
            channels=1
        )
        
        serialized = serialize_event(original_event)
        deserialized = deserialize_event(serialized)
        
        assert isinstance(deserialized, TTSAudioEvent)
        assert deserialized.audio_data is None  # Audio data should be removed
        assert deserialized.sample_rate == 16000
        assert deserialized.channels == 1
    
    def test_deserialize_event_with_timestamp_string(self):
        """Test deserializing event with string timestamp."""
        data = {
            'event_type': 'stt_transcript',
            'text': 'Test text',
            'timestamp': '2023-12-01T10:30:00',
            'event_id': 'test-id-123'
        }
        
        deserialized = deserialize_event(data)
        
        assert isinstance(deserialized, STTTranscriptEvent)
        assert deserialized.text == 'Test text'
        assert isinstance(deserialized.timestamp, datetime)
        assert deserialized.timestamp.year == 2023
        assert deserialized.timestamp.month == 12
    
    def test_deserialize_invalid_event_type(self):
        """Test deserializing with invalid event type."""
        data = {
            'event_type': 'invalid_event_type',
            'text': 'Test text'
        }
        
        with pytest.raises(ValueError, match="Unknown event type: invalid_event_type"):
            deserialize_event(data)
    
    def test_deserialize_missing_event_type(self):
        """Test deserializing with missing event type."""
        data = {
            'text': 'Test text'
        }
        
        with pytest.raises(ValueError, match="Event data missing event_type"):
            deserialize_event(data)
    
    def test_round_trip_serialization_all_event_types(self):
        """Test round-trip serialization for all major event types."""
        test_events = [
            # STT Events
            STTTranscriptEvent(text="Test transcript", confidence=0.9),
            STTPartialTranscriptEvent(text="Partial text"),
            STTErrorEvent(error=Exception("STT error")),
            STTConnectionEvent(connection_state=ConnectionState.CONNECTED),
            
            # TTS Events  
            TTSAudioEvent(audio_data=b"audio", sample_rate=16000),
            TTSSynthesisStartEvent(text="Synthesis text"),
            TTSSynthesisCompleteEvent(synthesis_id="synth-123"),
            TTSErrorEvent(error=Exception("TTS error")),
            TTSConnectionEvent(connection_state=ConnectionState.DISCONNECTED),
            
            # STS Events
            STSConnectedEvent(provider="test_provider"),
            STSDisconnectedEvent(reason="user_disconnect"),
            STSAudioInputEvent(audio_data=b"input"),
            STSAudioOutputEvent(audio_data=b"output"),
            STSTranscriptEvent(text="STS transcript"),
            STSResponseEvent(text="STS response"),
            STSConversationItemEvent(item_type="message"),
            STSErrorEvent(error=Exception("STS error")),
            
            # VAD Events
            VADSpeechStartEvent(speech_probability=0.8),
            VADSpeechEndEvent(total_speech_duration_ms=1000.0),
            VADAudioEvent(audio_data=b"vad audio"),
            VADPartialEvent(speech_probability=0.7),
            VADErrorEvent(error=Exception("VAD error")),
            
            # Generic Events
            PluginInitializedEvent(plugin_type="STT"),
            PluginClosedEvent(plugin_type="TTS"),
            PluginErrorEvent(plugin_type="VAD", error=Exception("Plugin error"))
        ]
        
        for original_event in test_events:
            # Serialize
            serialized = serialize_event(original_event)
            
            # Deserialize
            deserialized = deserialize_event(serialized)
            
            # Verify type and basic properties
            assert isinstance(deserialized, type(original_event))
            assert deserialized.event_type == original_event.event_type
            
            # Verify specific properties based on event type
            if hasattr(original_event, 'text') and original_event.text:
                assert deserialized.text == original_event.text
            
            if hasattr(original_event, 'error') and original_event.error:
                assert isinstance(deserialized.error, Exception)
                assert str(deserialized.error) == str(original_event.error)
    
    def test_create_event_function(self):
        """Test the create_event factory function."""
        # Test creating different event types
        stt_event = create_event(EventType.STT_TRANSCRIPT, text="Test text")
        assert isinstance(stt_event, STTTranscriptEvent)
        assert stt_event.text == "Test text"
        
        tts_event = create_event(EventType.TTS_AUDIO, sample_rate=22050)
        assert isinstance(tts_event, TTSAudioEvent)
        assert tts_event.sample_rate == 22050
        
        vad_event = create_event(EventType.VAD_SPEECH_START, speech_probability=0.9)
        assert isinstance(vad_event, VADSpeechStartEvent)
        assert vad_event.speech_probability == 0.9
    
    def test_create_event_invalid_type(self):
        """Test create_event with invalid event type."""
        # Test with None (which should raise an error)
        with pytest.raises((ValueError, TypeError)):
            create_event(None, text="test")
    
    def test_json_round_trip_serialization(self):
        """Test complete JSON round-trip serialization."""
        events = [
            STTTranscriptEvent(text="First", confidence=0.95),
            STTErrorEvent(error=ValueError("Test error"), error_code="E001"),
            TTSAudioEvent(audio_data=b"audio data", sample_rate=16000)
        ]
        
        # Serialize to JSON string
        json_string = serialize_events(events)
        
        # Parse JSON and deserialize each event
        events_data = json.loads(json_string)
        deserialized_events = [deserialize_event(event_data) for event_data in events_data]
        
        # Verify we got the same number and types of events
        assert len(deserialized_events) == 3
        assert isinstance(deserialized_events[0], STTTranscriptEvent)
        assert isinstance(deserialized_events[1], STTErrorEvent)
        assert isinstance(deserialized_events[2], TTSAudioEvent)
        
        # Verify content
        assert deserialized_events[0].text == "First"
        assert deserialized_events[0].confidence == 0.95
        assert str(deserialized_events[1].error) == "Test error"
        assert deserialized_events[1].error_code == "E001"
        assert deserialized_events[2].sample_rate == 16000
        assert deserialized_events[2].audio_data is None  # Should be removed during serialization


class TestEventFiltering:
    """Test event filtering functionality."""
    
    def test_event_filter_creation(self):
        """Test EventFilter creation with various parameters."""
        from getstream.plugins.common.event_utils import EventFilter
        
        # Test basic filter
        filter1 = EventFilter()
        assert filter1.event_types is None
        assert filter1.session_ids is None
        assert filter1.plugin_names is None
        assert filter1.time_window_ms is None
        assert filter1.min_confidence is None
        
        # Test filter with specific criteria
        filter2 = EventFilter(
            event_types=[EventType.STT_TRANSCRIPT, EventType.STT_PARTIAL_TRANSCRIPT],
            session_ids=["session1", "session2"],
            plugin_names=["stt_plugin"],
            time_window_ms=60000,
            min_confidence=0.8
        )
        assert EventType.STT_TRANSCRIPT in filter2.event_types
        assert EventType.STT_PARTIAL_TRANSCRIPT in filter2.event_types
        assert "session1" in filter2.session_ids
        assert "session2" in filter2.session_ids
        assert "stt_plugin" in filter2.plugin_names
        assert filter2.time_window_ms == 60000
        assert filter2.min_confidence == 0.8
    
    def test_event_filter_matching(self):
        """Test EventFilter.matches() method."""
        from getstream.plugins.common.event_utils import EventFilter
        from datetime import datetime, timedelta
        
        # Create test events
        now = datetime.now()
        event1 = STTTranscriptEvent(
            text="test",
            confidence=0.9,
            session_id="session1",
            plugin_name="stt_plugin",
            timestamp=now
        )
        event2 = STTTranscriptEvent(
            text="test2",
            confidence=0.7,
            session_id="session2",
            plugin_name="tts_plugin",
            timestamp=now - timedelta(seconds=120)
        )
        
        # Test event type filtering
        type_filter = EventFilter(event_types=[EventType.STT_TRANSCRIPT])
        assert type_filter.matches(event1) is True
        assert type_filter.matches(event2) is True
        
        type_filter = EventFilter(event_types=[EventType.TTS_AUDIO])
        assert type_filter.matches(event1) is False
        assert type_filter.matches(event2) is False
        
        # Test session ID filtering
        session_filter = EventFilter(session_ids=["session1"])
        assert session_filter.matches(event1) is True
        assert session_filter.matches(event2) is False
        
        # Test plugin name filtering
        plugin_filter = EventFilter(plugin_names=["stt_plugin"])
        assert plugin_filter.matches(event1) is True
        assert plugin_filter.matches(event2) is False
        
        # Test time window filtering
        time_filter = EventFilter(time_window_ms=60000)  # Last minute
        assert time_filter.matches(event1) is True  # Recent
        assert time_filter.matches(event2) is False  # 2 minutes ago
        
        # Test confidence filtering
        confidence_filter = EventFilter(min_confidence=0.8)
        assert confidence_filter.matches(event1) is True  # 0.9 > 0.8
        assert confidence_filter.matches(event2) is False  # 0.7 < 0.8
        
        # Test combined filtering
        combined_filter = EventFilter(
            event_types=[EventType.STT_TRANSCRIPT],
            session_ids=["session1"],
            plugin_names=["stt_plugin"],
            min_confidence=0.8
        )
        assert combined_filter.matches(event1) is True  # All criteria match
        assert combined_filter.matches(event2) is False  # Some criteria don't match
    
    def test_event_filter_edge_cases(self):
        """Test EventFilter edge cases."""
        from getstream.plugins.common.event_utils import EventFilter
        
        # Test with None values
        event = STTTranscriptEvent(text="test")
        filter_none = EventFilter()
        assert filter_none.matches(event) is True  # No filters = match everything
        
        # Test with empty sets (current implementation treats empty sets as "no filter")
        filter_empty = EventFilter(event_types=[], session_ids=[], plugin_names=[])
        assert filter_empty.matches(event) is True  # Empty sets = no filtering = match everything
        
        # Test with None event (current implementation doesn't validate event parameter)
        filter_any = EventFilter()
        assert filter_any.matches(None) is True  # None events currently pass through (could be improved)


class TestEventRegistry:
    """Test event registry functionality."""
    
    def test_event_registry_creation(self):
        """Test EventRegistry creation and basic properties."""
        from getstream.plugins.common.event_utils import EventRegistry
        
        registry = EventRegistry(max_events=100)
        assert registry.max_events == 100
        assert len(registry.events) == 0
        assert len(registry.event_counts) == 0
        assert len(registry.session_events) == 0
        assert len(registry.error_counts) == 0
    
    def test_event_registration(self):
        """Test event registration and counting."""
        from getstream.plugins.common.event_utils import EventRegistry
        
        registry = EventRegistry()
        
        # Register some events
        event1 = STTTranscriptEvent(text="test1", session_id="session1")
        event2 = STTTranscriptEvent(text="test2", session_id="session1")
        event3 = STTErrorEvent(error=Exception("test"), session_id="session2")
        
        registry.register_event(event1)
        registry.register_event(event2)
        registry.register_event(event3)
        
        # Check counts
        assert len(registry.events) == 3
        assert registry.event_counts[EventType.STT_TRANSCRIPT] == 2
        assert registry.event_counts[EventType.STT_ERROR] == 1
        
        # Check session grouping
        assert len(registry.session_events["session1"]) == 2
        assert len(registry.session_events["session2"]) == 1
        
        # Check error counting
        assert registry.error_counts["None_stt_error"] == 1
    
    def test_event_registry_max_events(self):
        """Test EventRegistry respects max_events limit."""
        from getstream.plugins.common.event_utils import EventRegistry
        
        registry = EventRegistry(max_events=3)
        
        # Add 5 events
        for i in range(5):
            event = STTTranscriptEvent(text=f"test{i}")
            registry.register_event(event)
        
        # Should only keep the last 3
        assert len(registry.events) == 3
        assert registry.events[0].text == "test2"  # First event should be dropped
        assert registry.events[2].text == "test4"  # Last event should be kept
    
    def test_event_registry_filtering(self):
        """Test EventRegistry.get_events() with filtering."""
        from getstream.plugins.common.event_utils import EventRegistry, EventFilter
        
        registry = EventRegistry()
        
        # Add events with different characteristics
        events = [
            STTTranscriptEvent(text="high_conf", confidence=0.95, session_id="session1"),
            STTTranscriptEvent(text="low_conf", confidence=0.75, session_id="session1"),
            STTTranscriptEvent(text="other_session", confidence=0.9, session_id="session2"),
            TTSAudioEvent(audio_data=b"audio", session_id="session1")
        ]
        
        for event in events:
            registry.register_event(event)
        
        # Test filtering by event type
        stt_filter = EventFilter(event_types=[EventType.STT_TRANSCRIPT])
        stt_events = registry.get_events(stt_filter)
        assert len(stt_events) == 3
        assert all(e.event_type == EventType.STT_TRANSCRIPT for e in stt_events)
        
        # Test filtering by session
        session_filter = EventFilter(session_ids=["session1"])
        session_events = registry.get_events(session_filter)
        assert len(session_events) == 3
        assert all(e.session_id == "session1" for e in session_events)
        
        # Test filtering by confidence (strict filtering - events without confidence are excluded)
        confidence_filter = EventFilter(min_confidence=0.8)
        high_conf_events = registry.get_events(confidence_filter)
        assert len(high_conf_events) == 2  # Only STT events with confidence >= 0.8, TTSAudioEvent excluded
        # Verify all returned events meet the confidence threshold
        assert all(e.confidence >= 0.8 for e in high_conf_events)
        
        # Test combined filtering
        combined_filter = EventFilter(
            event_types=[EventType.STT_TRANSCRIPT],
            session_ids=["session1"],
            min_confidence=0.8
        )
        combined_events = registry.get_events(combined_filter)
        assert len(combined_events) == 1
        assert combined_events[0].text == "high_conf"
    
    def test_event_registry_statistics(self):
        """Test EventRegistry statistics methods."""
        from getstream.plugins.common.event_utils import EventRegistry
        
        registry = EventRegistry()
        
        # Add events
        events = [
            STTTranscriptEvent(text="test1", session_id="session1"),
            STTTranscriptEvent(text="test2", session_id="session1"),
            STTErrorEvent(error=Exception("error1"), session_id="session2"),
            TTSAudioEvent(audio_data=b"audio", session_id="session3")
        ]
        
        for event in events:
            registry.register_event(event)
        
        # Test error summary
        error_summary = registry.get_error_summary()
        assert "None_stt_error" in error_summary["error_breakdown"]
        assert error_summary["error_breakdown"]["None_stt_error"] == 1
        
        # Test statistics
        stats = registry.get_statistics()
        assert stats["total_events"] == 4
        assert stats["session_count"] == 3
        assert stats["event_distribution"]["stt_transcript"] == 2
        assert stats["event_distribution"]["stt_error"] == 1
        assert stats["event_distribution"]["tts_audio"] == 1
    
    def test_event_registry_listeners(self):
        """Test EventRegistry event listener functionality."""
        from getstream.plugins.common.event_utils import EventRegistry
        
        registry = EventRegistry()
        received_events = []
        
        def listener(event):
            received_events.append(event)
        
        # Add listener for STT events
        registry.add_listener(EventType.STT_TRANSCRIPT, listener)
        
        # Register an event
        event = STTTranscriptEvent(text="test")
        registry.register_event(event)
        
        # Check if listener was called
        assert len(received_events) == 1
        assert received_events[0] == event
        
        # Remove listener
        registry.remove_listener(EventType.STT_TRANSCRIPT, listener)
        received_events.clear()
        
        # Register another event
        event2 = STTTranscriptEvent(text="test2")
        registry.register_event(event2)
        
        # Listener should not be called
        assert len(received_events) == 0
    
    def test_event_registry_clear(self):
        """Test EventRegistry.clear() method."""
        from getstream.plugins.common.event_utils import EventRegistry
        
        registry = EventRegistry()
        
        # Add some events
        event = STTTranscriptEvent(text="test")
        registry.register_event(event)
        
        # Verify events exist
        assert len(registry.events) == 1
        assert len(registry.event_counts) > 0
        
        # Clear registry
        registry.clear()
        
        # Verify everything is cleared
        assert len(registry.events) == 0
        assert len(registry.event_counts) == 0
        assert len(registry.session_events) == 0
        assert len(registry.error_counts) == 0


class TestEventLogger:
    """Test event logging functionality."""
    
    def test_event_logger_creation(self):
        """Test EventLogger creation."""
        from getstream.plugins.common.event_utils import EventLogger
        
        logger = EventLogger()
        assert logger.registry is not None
        assert logger.logger is not None
    
    def test_event_logger_logging(self):
        """Test EventLogger event logging."""
        from getstream.plugins.common.event_utils import EventLogger
        import logging
        
        # Set up logging capture
        log_records = []
        
        def capture_logs(record):
            log_records.append(record)
        
        # Create logger and capture logs
        logger = EventLogger()
        logger.logger.handlers = []  # Remove default handlers
        logger.logger.addHandler(logging.StreamHandler())
        
        # Log some events
        event1 = STTTranscriptEvent(text="test1")
        event2 = STTErrorEvent(error=Exception("test error"))
        
        logger.log_event(event1)
        logger.log_event(event2)
        
        # Check that events were registered
        assert len(logger.registry.events) == 2
        
        # Check that events were logged (basic verification)
        assert len(logger.registry.events) == 2
    
    def test_event_logger_batch_logging(self):
        """Test EventLogger batch logging."""
        from getstream.plugins.common.event_utils import EventLogger
        
        logger = EventLogger()
        
        # Create multiple events
        events = [
            STTTranscriptEvent(text="test1"),
            STTTranscriptEvent(text="test2"),
            TTSAudioEvent(audio_data=b"audio")
        ]
        
        # Log events individually (EventLogger doesn't have batch method)
        for event in events:
            logger.log_event(event)
        
        # Check all events were registered
        assert len(logger.registry.events) == 3
        
        # Check event types
        event_types = [e.event_type for e in logger.registry.events]
        assert EventType.STT_TRANSCRIPT in event_types
        assert EventType.TTS_AUDIO in event_types


class TestGlobalEventSystem:
    """Test global event system functionality."""
    
    def test_global_registry_access(self):
        """Test global registry access functions."""
        from getstream.plugins.common.event_utils import (
            get_global_registry, get_global_logger, register_global_event
        )
        
        # Get global instances
        registry = get_global_registry()
        logger = get_global_logger()
        
        assert registry is not None
        assert logger is not None
        
        # Test global event registration
        event = STTTranscriptEvent(text="global_test")
        register_global_event(event)
        
        # Check event was registered
        global_events = registry.get_events()
        assert len(global_events) > 0
        
        # Find our event - filter by event type and text to avoid issues with other event types
        test_events = [e for e in global_events if hasattr(e, 'text') and e.text == "global_test"]
        assert len(test_events) == 1
    
    def test_global_event_consistency(self):
        """Test that global registry and logger are consistent."""
        from getstream.plugins.common.event_utils import (
            get_global_registry, get_global_logger, register_global_event
        )
        
        registry = get_global_registry()
        logger = get_global_logger()
        
        # They are separate instances but register_global_event ensures consistency
        assert logger.registry is not registry  # Separate instances
        
        # Test consistency through global registration
        event = STTTranscriptEvent(text="consistency_test")
        register_global_event(event)
        
        # Check both registries have the event
        registry_events = registry.get_events()
        logger_events = logger.registry.get_events()
        
        registry_test_events = [e for e in registry_events if hasattr(e, 'text') and e.text == "consistency_test"]
        logger_test_events = [e for e in logger_events if hasattr(e, 'text') and e.text == "consistency_test"]
        
        assert len(registry_test_events) == 1
        assert len(logger_test_events) == 1
