"""Tests for MetricsCollector handler methods.

These tests verify that the MetricsCollector correctly records metrics
when handling various events. Since the EventManager requires a running
event loop and complex type resolution, we test the handler methods directly.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from unittest.mock import MagicMock

from vision_agents.core.llm.events import (
    LLMResponseCompletedEvent,
    ToolEndEvent,
    VLMInferenceCompletedEvent,
    VLMErrorEvent,
)
from vision_agents.core.stt.events import STTTranscriptEvent, STTErrorEvent, TranscriptResponse
from vision_agents.core.tts.events import TTSSynthesisCompleteEvent, TTSErrorEvent
from vision_agents.core.turn_detection.events import TurnEndedEvent


@dataclass
class MockMetric:
    """Mock metric that records all calls."""

    name: str
    calls: List[tuple] = field(default_factory=list)

    def record(self, value: float, attributes: Optional[Dict] = None):
        self.calls.append(("record", value, attributes or {}))

    def add(self, value: int, attributes: Optional[Dict] = None):
        self.calls.append(("add", value, attributes or {}))


class TestMetricsCollectorHandlers:
    """Tests for MetricsCollector handler methods."""

    def _create_collector_with_mocks(self, monkeypatch):
        """Create a collector with mocked metrics."""
        # Create mock metrics
        mocks = {
            "llm_latency_ms": MockMetric("llm_latency_ms"),
            "llm_time_to_first_token_ms": MockMetric("llm_time_to_first_token_ms"),
            "llm_input_tokens": MockMetric("llm_input_tokens"),
            "llm_output_tokens": MockMetric("llm_output_tokens"),
            "llm_errors": MockMetric("llm_errors"),
            "llm_tool_calls": MockMetric("llm_tool_calls"),
            "llm_tool_latency_ms": MockMetric("llm_tool_latency_ms"),
            "stt_latency_ms": MockMetric("stt_latency_ms"),
            "stt_audio_duration_ms": MockMetric("stt_audio_duration_ms"),
            "stt_errors": MockMetric("stt_errors"),
            "tts_latency_ms": MockMetric("tts_latency_ms"),
            "tts_audio_duration_ms": MockMetric("tts_audio_duration_ms"),
            "tts_characters": MockMetric("tts_characters"),
            "tts_errors": MockMetric("tts_errors"),
            "turn_duration_ms": MockMetric("turn_duration_ms"),
            "turn_trailing_silence_ms": MockMetric("turn_trailing_silence_ms"),
            "vlm_inferences": MockMetric("vlm_inferences"),
            "vlm_inference_latency_ms": MockMetric("vlm_inference_latency_ms"),
            "vlm_input_tokens": MockMetric("vlm_input_tokens"),
            "vlm_output_tokens": MockMetric("vlm_output_tokens"),
            "vlm_errors": MockMetric("vlm_errors"),
            "video_frames_processed": MockMetric("video_frames_processed"),
            "video_detections": MockMetric("video_detections"),
        }

        # Patch all metrics
        for name, mock in mocks.items():
            monkeypatch.setattr(
                f"vision_agents.core.observability.collector.metrics.{name}",
                mock,
            )

        # Create a minimal collector without subscribing to events
        from vision_agents.core.observability.collector import MetricsCollector

        # Create a mock agent
        agent = MagicMock()
        agent.llm = None
        agent.stt = None
        agent.tts = None
        agent.turn_detection = None
        agent.events = MagicMock()

        # Create collector but skip event subscription
        collector = object.__new__(MetricsCollector)
        collector.agent = agent
        collector._realtime_session_starts = {}

        return collector, mocks

    def test_on_llm_response_completed(self, monkeypatch):
        """Test LLM response completed handler records all metrics."""
        collector, mocks = self._create_collector_with_mocks(monkeypatch)

        event = LLMResponseCompletedEvent(
            plugin_name="openai",
            text="Hello",
            latency_ms=150.0,
            time_to_first_token_ms=50.0,
            input_tokens=10,
            output_tokens=5,
            model="gpt-4",
        )

        collector._on_llm_response_completed(event)

        assert len(mocks["llm_latency_ms"].calls) == 1
        assert mocks["llm_latency_ms"].calls[0] == (
            "record",
            150.0,
            {"provider": "openai", "model": "gpt-4"},
        )

        assert len(mocks["llm_time_to_first_token_ms"].calls) == 1
        assert mocks["llm_time_to_first_token_ms"].calls[0] == (
            "record",
            50.0,
            {"provider": "openai", "model": "gpt-4"},
        )

        assert len(mocks["llm_input_tokens"].calls) == 1
        assert mocks["llm_input_tokens"].calls[0] == (
            "add",
            10,
            {"provider": "openai", "model": "gpt-4"},
        )

        assert len(mocks["llm_output_tokens"].calls) == 1
        assert mocks["llm_output_tokens"].calls[0] == (
            "add",
            5,
            {"provider": "openai", "model": "gpt-4"},
        )

    def test_on_llm_response_completed_partial_data(self, monkeypatch):
        """Test LLM handler with missing optional fields."""
        collector, mocks = self._create_collector_with_mocks(monkeypatch)

        event = LLMResponseCompletedEvent(
            plugin_name="openai",
            text="Hello",
            # No latency, tokens, or model
        )

        collector._on_llm_response_completed(event)

        # Should not record metrics for missing fields
        assert len(mocks["llm_latency_ms"].calls) == 0
        assert len(mocks["llm_time_to_first_token_ms"].calls) == 0
        assert len(mocks["llm_input_tokens"].calls) == 0
        assert len(mocks["llm_output_tokens"].calls) == 0

    def test_on_tool_end(self, monkeypatch):
        """Test tool end handler records metrics."""
        collector, mocks = self._create_collector_with_mocks(monkeypatch)

        event = ToolEndEvent(
            plugin_name="openai",
            tool_name="get_weather",
            success=True,
            execution_time_ms=25.0,
        )

        collector._on_tool_end(event)

        assert len(mocks["llm_tool_calls"].calls) == 1
        assert mocks["llm_tool_calls"].calls[0] == (
            "add",
            1,
            {"provider": "openai", "tool_name": "get_weather", "success": "true"},
        )

        assert len(mocks["llm_tool_latency_ms"].calls) == 1
        assert mocks["llm_tool_latency_ms"].calls[0] == (
            "record",
            25.0,
            {"provider": "openai", "tool_name": "get_weather", "success": "true"},
        )

    def test_on_stt_transcript(self, monkeypatch):
        """Test STT transcript handler records metrics."""
        collector, mocks = self._create_collector_with_mocks(monkeypatch)

        event = STTTranscriptEvent(
            plugin_name="deepgram",
            text="Hello world",
            response=TranscriptResponse(
                processing_time_ms=100.0,
                audio_duration_ms=2000.0,
                model_name="nova-2",
                language="en",
            ),
        )

        collector._on_stt_transcript(event)

        assert len(mocks["stt_latency_ms"].calls) == 1
        assert mocks["stt_latency_ms"].calls[0] == (
            "record",
            100.0,
            {"provider": "deepgram", "model": "nova-2", "language": "en"},
        )

        assert len(mocks["stt_audio_duration_ms"].calls) == 1
        assert mocks["stt_audio_duration_ms"].calls[0] == (
            "record",
            2000.0,
            {"provider": "deepgram", "model": "nova-2", "language": "en"},
        )

    def test_on_stt_error(self, monkeypatch):
        """Test STT error handler records metrics."""
        collector, mocks = self._create_collector_with_mocks(monkeypatch)

        event = STTErrorEvent(
            plugin_name="deepgram",
            error=ValueError("Connection failed"),
            error_code="CONNECTION_ERROR",
        )

        collector._on_stt_error(event)

        assert len(mocks["stt_errors"].calls) == 1
        assert mocks["stt_errors"].calls[0][0] == "add"
        assert mocks["stt_errors"].calls[0][1] == 1
        assert mocks["stt_errors"].calls[0][2]["provider"] == "deepgram"
        assert mocks["stt_errors"].calls[0][2]["error_type"] == "ValueError"
        assert mocks["stt_errors"].calls[0][2]["error_code"] == "CONNECTION_ERROR"

    def test_on_tts_synthesis_complete(self, monkeypatch):
        """Test TTS synthesis complete handler records metrics."""
        collector, mocks = self._create_collector_with_mocks(monkeypatch)

        event = TTSSynthesisCompleteEvent(
            plugin_name="cartesia",
            text="Hello world",
            synthesis_time_ms=50.0,
            audio_duration_ms=1500.0,
        )

        collector._on_tts_synthesis_complete(event)

        assert len(mocks["tts_latency_ms"].calls) == 1
        assert mocks["tts_latency_ms"].calls[0] == ("record", 50.0, {"provider": "cartesia"})

        assert len(mocks["tts_audio_duration_ms"].calls) == 1
        assert mocks["tts_audio_duration_ms"].calls[0] == (
            "record",
            1500.0,
            {"provider": "cartesia"},
        )

        assert len(mocks["tts_characters"].calls) == 1
        assert mocks["tts_characters"].calls[0] == (
            "add",
            11,
            {"provider": "cartesia"},
        )  # len("Hello world")

    def test_on_tts_error(self, monkeypatch):
        """Test TTS error handler records metrics."""
        collector, mocks = self._create_collector_with_mocks(monkeypatch)

        event = TTSErrorEvent(
            plugin_name="cartesia",
            error=RuntimeError("Synthesis failed"),
            error_code="SYNTHESIS_ERROR",
        )

        collector._on_tts_error(event)

        assert len(mocks["tts_errors"].calls) == 1
        assert mocks["tts_errors"].calls[0][0] == "add"
        assert mocks["tts_errors"].calls[0][1] == 1
        assert mocks["tts_errors"].calls[0][2]["provider"] == "cartesia"
        assert mocks["tts_errors"].calls[0][2]["error_type"] == "RuntimeError"

    def test_on_turn_ended(self, monkeypatch):
        """Test turn ended handler records metrics."""
        collector, mocks = self._create_collector_with_mocks(monkeypatch)

        event = TurnEndedEvent(
            plugin_name="smart_turn",
            duration_ms=3500.0,
            trailing_silence_ms=500.0,
        )

        collector._on_turn_ended(event)

        assert len(mocks["turn_duration_ms"].calls) == 1
        assert mocks["turn_duration_ms"].calls[0] == (
            "record",
            3500.0,
            {"provider": "smart_turn"},
        )

        assert len(mocks["turn_trailing_silence_ms"].calls) == 1
        assert mocks["turn_trailing_silence_ms"].calls[0] == (
            "record",
            500.0,
            {"provider": "smart_turn"},
        )

    def test_on_vlm_inference_completed(self, monkeypatch):
        """Test VLM inference completed handler records metrics."""
        collector, mocks = self._create_collector_with_mocks(monkeypatch)

        event = VLMInferenceCompletedEvent(
            plugin_name="moondream",
            model="moondream-cloud",
            text="A person walking",
            latency_ms=200.0,
            frames_processed=5,
            input_tokens=100,
            output_tokens=20,
        )

        collector._on_vlm_inference_completed(event)

        assert len(mocks["vlm_inferences"].calls) == 1
        assert mocks["vlm_inferences"].calls[0] == (
            "add",
            1,
            {"provider": "moondream", "model": "moondream-cloud"},
        )

        assert len(mocks["vlm_inference_latency_ms"].calls) == 1
        assert mocks["vlm_inference_latency_ms"].calls[0] == (
            "record",
            200.0,
            {"provider": "moondream", "model": "moondream-cloud"},
        )

        assert len(mocks["video_frames_processed"].calls) == 1
        assert mocks["video_frames_processed"].calls[0] == (
            "add",
            5,
            {"provider": "moondream", "model": "moondream-cloud"},
        )

        assert len(mocks["vlm_input_tokens"].calls) == 1
        assert mocks["vlm_input_tokens"].calls[0] == (
            "add",
            100,
            {"provider": "moondream", "model": "moondream-cloud"},
        )

        assert len(mocks["vlm_output_tokens"].calls) == 1
        assert mocks["vlm_output_tokens"].calls[0] == (
            "add",
            20,
            {"provider": "moondream", "model": "moondream-cloud"},
        )

    def test_on_vlm_error(self, monkeypatch):
        """Test VLM error handler records metrics."""
        collector, mocks = self._create_collector_with_mocks(monkeypatch)

        event = VLMErrorEvent(
            plugin_name="moondream",
            error=RuntimeError("Inference failed"),
            error_code="INFERENCE_ERROR",
        )

        collector._on_vlm_error(event)

        assert len(mocks["vlm_errors"].calls) == 1
        assert mocks["vlm_errors"].calls[0][0] == "add"
        assert mocks["vlm_errors"].calls[0][1] == 1
        assert mocks["vlm_errors"].calls[0][2]["provider"] == "moondream"
        assert mocks["vlm_errors"].calls[0][2]["error_type"] == "RuntimeError"

    def test_base_attributes_extracts_provider(self, monkeypatch):
        """Test that base attributes correctly extracts provider."""
        collector, _ = self._create_collector_with_mocks(monkeypatch)

        event = LLMResponseCompletedEvent(
            plugin_name="test_provider",
            text="Hello",
        )

        attrs = collector._base_attributes(event)
        assert attrs == {"provider": "test_provider"}

    def test_base_attributes_handles_missing_plugin_name(self, monkeypatch):
        """Test that base attributes handles missing plugin_name."""
        collector, _ = self._create_collector_with_mocks(monkeypatch)

        event = LLMResponseCompletedEvent(
            text="Hello",
        )

        attrs = collector._base_attributes(event)
        assert attrs == {}
