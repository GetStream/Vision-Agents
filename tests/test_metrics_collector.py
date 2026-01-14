"""Tests for MetricsCollector handler methods.

These tests verify that the MetricsCollector correctly records metrics
when handling various events. Since the EventManager requires a running
event loop and complex type resolution, we test the handler methods directly.
"""

from unittest.mock import MagicMock, patch

import pytest
from vision_agents.core.events import EventManager
from vision_agents.core.llm.events import (
    LLMResponseCompletedEvent,
    ToolEndEvent,
    VLMErrorEvent,
    VLMInferenceCompletedEvent,
)
from vision_agents.core.observability.agent import AgentMetrics
from vision_agents.core.observability.collector import MetricsCollector
from vision_agents.core.observability.metrics import (
    llm_errors,
    llm_input_tokens,
    llm_latency_ms,
    llm_output_tokens,
    llm_time_to_first_token_ms,
    llm_tool_calls,
    llm_tool_latency_ms,
    meter,
    stt_audio_duration_ms,
    stt_errors,
    stt_latency_ms,
    tts_audio_duration_ms,
    tts_characters,
    tts_errors,
    tts_latency_ms,
    turn_duration_ms,
    turn_trailing_silence_ms,
    video_detections,
    video_frames_processed,
    vlm_errors,
    vlm_inference_latency_ms,
    vlm_inferences,
    vlm_input_tokens,
    vlm_output_tokens,
)
from vision_agents.core.stt.events import (
    STTErrorEvent,
    STTTranscriptEvent,
    TranscriptResponse,
)
from vision_agents.core.tts.events import TTSErrorEvent, TTSSynthesisCompleteEvent
from vision_agents.core.turn_detection.events import TurnEndedEvent


@pytest.fixture()
def mock_metrics():
    """
    Go over all the used metrics and patch their methods to record the calls.
    """
    all_metrics = [
        llm_errors,
        llm_input_tokens,
        llm_latency_ms,
        llm_output_tokens,
        llm_time_to_first_token_ms,
        llm_tool_calls,
        llm_tool_latency_ms,
        meter,
        stt_audio_duration_ms,
        stt_errors,
        stt_latency_ms,
        tts_audio_duration_ms,
        tts_characters,
        tts_errors,
        tts_latency_ms,
        turn_duration_ms,
        turn_trailing_silence_ms,
        video_detections,
        video_frames_processed,
        vlm_errors,
        vlm_inference_latency_ms,
        vlm_inferences,
        vlm_input_tokens,
        vlm_output_tokens,
    ]
    patches = []
    try:
        for metric in all_metrics:
            if hasattr(metric, "record"):
                patches.append(patch.object(metric, "record").start())
            if hasattr(metric, "add"):
                patches.append(patch.object(metric, "add").start())
        yield
    finally:
        for patch_ in reversed(patches):
            patch_.stop()


@pytest.fixture
async def event_manager() -> EventManager:
    manager = EventManager()

    events = [
        LLMResponseCompletedEvent,
        STTErrorEvent,
        STTTranscriptEvent,
        TTSErrorEvent,
        TTSSynthesisCompleteEvent,
        ToolEndEvent,
        TurnEndedEvent,
        VLMErrorEvent,
        VLMInferenceCompletedEvent,
    ]
    for cls in events:
        manager.register(cls)
    return manager


@pytest.fixture
async def agent(event_manager):
    agent = MagicMock()
    agent.llm = MagicMock()
    agent.llm.events = event_manager

    agent.stt = MagicMock()
    agent.stt.events = event_manager

    agent.tts = MagicMock()
    agent.tts.events = event_manager

    agent.turn_detection = MagicMock()
    agent.turn_detection.events = event_manager
    agent.metrics = AgentMetrics()
    agent.events = EventManager()
    return agent


@pytest.fixture
async def collector(mock_metrics, agent) -> MetricsCollector:
    collector = MetricsCollector(agent)
    return collector


class TestMetricsCollector:
    """Tests for MetricsCollector handler methods."""

    async def test_on_llm_response_completed(self, collector, event_manager, agent):
        """Test LLM response completed handler records all metrics."""

        event = LLMResponseCompletedEvent(
            plugin_name="openai",
            text="Hello",
            latency_ms=150.0,
            time_to_first_token_ms=50.0,
            input_tokens=10,
            output_tokens=5,
            model="gpt-4",
        )
        event_manager.send(event)
        await event_manager.wait(1)

        llm_latency_ms.record.assert_called_once_with(
            150.0, {"provider": "openai", "model": "gpt-4"}
        )
        llm_time_to_first_token_ms.record.assert_called_once_with(
            50.0, {"provider": "openai", "model": "gpt-4"}
        )
        llm_input_tokens.add.assert_called_once_with(
            10, {"provider": "openai", "model": "gpt-4"}
        )
        llm_output_tokens.add.assert_called_once_with(
            5, {"provider": "openai", "model": "gpt-4"}
        )
        assert agent.metrics.llm_latency_ms__avg.value() == 150
        assert agent.metrics.llm_time_to_first_token_ms__avg.value() == 50
        assert agent.metrics.llm_input_tokens__total.value() == 10
        assert agent.metrics.llm_output_tokens__total.value() == 5

    async def test_on_llm_response_completed_partial_data(
        self, collector, event_manager, agent
    ):
        """Test LLM handler with missing optional fields."""

        event = LLMResponseCompletedEvent(
            plugin_name="openai",
            text="Hello",
            # No latency, tokens, or model
        )

        event_manager.send(event)
        await event_manager.wait(1)

        # Should not record metrics for missing fields
        llm_latency_ms.record.assert_not_called()
        llm_time_to_first_token_ms.record.assert_not_called()
        llm_input_tokens.add.assert_not_called()
        llm_output_tokens.add.assert_not_called()

        assert agent.metrics.llm_latency_ms__avg.value() is None
        assert agent.metrics.llm_time_to_first_token_ms__avg.value() is None
        assert agent.metrics.llm_input_tokens__total.value() == 0
        assert agent.metrics.llm_output_tokens__total.value() == 0

    async def test_on_tool_end(self, collector, event_manager, agent):
        """Test tool end handler records metrics."""

        event = ToolEndEvent(
            plugin_name="openai",
            tool_name="get_weather",
            success=True,
            execution_time_ms=25.0,
        )

        event_manager.send(event)
        await event_manager.wait(1)

        llm_tool_calls.add.assert_called_once_with(
            1, {"provider": "openai", "tool_name": "get_weather", "success": "true"}
        )
        llm_tool_latency_ms.record.assert_called_once_with(
            25.0, {"provider": "openai", "tool_name": "get_weather", "success": "true"}
        )

        assert agent.metrics.llm_tool_calls__total.value() == 1
        assert agent.metrics.llm_tool_latency_ms__avg.value() == 25

    async def test_on_stt_transcript(self, collector, event_manager, agent):
        """Test STT transcript handler records metrics."""

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

        event_manager.send(event)
        await event_manager.wait(1)

        stt_latency_ms.record.assert_called_once_with(
            100.0, {"provider": "deepgram", "model": "nova-2", "language": "en"}
        )
        stt_audio_duration_ms.record.assert_called_once_with(
            2000.0,
            {"provider": "deepgram", "model": "nova-2", "language": "en"},
        )

        assert agent.metrics.stt_latency_ms__avg.value() == 100.0
        assert agent.metrics.stt_audio_duration_ms__total.value() == 2000.0

    async def test_on_stt_error(self, collector, event_manager):
        """Test STT error handler records metrics."""

        event = STTErrorEvent(
            plugin_name="deepgram",
            error=ValueError("Connection failed"),
            error_code="CONNECTION_ERROR",
        )

        event_manager.send(event)
        await event_manager.wait(1)

        stt_errors.add.assert_called_once_with(
            1,
            {
                "provider": "deepgram",
                "error_type": "ValueError",
                "error_code": "CONNECTION_ERROR",
            },
        )

    async def test_on_tts_synthesis_complete(self, collector, event_manager, agent):
        """Test TTS synthesis complete handler records metrics."""

        event = TTSSynthesisCompleteEvent(
            plugin_name="cartesia",
            text="Hello world",
            synthesis_time_ms=50.0,
            audio_duration_ms=1500.0,
        )

        event_manager.send(event)
        await event_manager.wait(1)

        tts_latency_ms.record.assert_called_once_with(50.0, {"provider": "cartesia"})
        tts_audio_duration_ms.record.assert_called_once_with(
            1500.0, {"provider": "cartesia"}
        )
        tts_characters.add.assert_called_once_with(
            len("Hello world"), {"provider": "cartesia"}
        )

        assert agent.metrics.tts_latency_ms__avg.value() == 50.0
        assert agent.metrics.tts_audio_duration_ms__total.value() == 1500.0
        assert agent.metrics.tts_characters__total.value() == len("Hello world")

    async def test_on_tts_error(self, collector, event_manager):
        """Test TTS error handler records metrics."""

        event = TTSErrorEvent(
            plugin_name="cartesia",
            error=RuntimeError("Synthesis failed"),
            error_code="SYNTHESIS_ERROR",
        )

        event_manager.send(event)
        await event_manager.wait(1)
        tts_errors.add.assert_called_once_with(
            1,
            {
                "provider": "cartesia",
                "error_type": "RuntimeError",
                "error_code": "SYNTHESIS_ERROR",
            },
        )

    async def test_on_turn_ended(self, collector, event_manager, agent):
        """Test turn ended handler records metrics."""

        event = TurnEndedEvent(
            plugin_name="smart_turn",
            duration_ms=3500.0,
            trailing_silence_ms=500.0,
        )

        event_manager.send(event)
        await event_manager.wait(1)

        turn_duration_ms.record.assert_called_once_with(
            3500.0, {"provider": "smart_turn"}
        )
        turn_trailing_silence_ms.record.assert_called_once_with(
            500.0, {"provider": "smart_turn"}
        )
        assert agent.metrics.turn_duration_ms__avg.value() == 3500.0
        assert agent.metrics.turn_trailing_silence_ms__avg.value() == 500.0

    async def test_on_vlm_inference_completed(self, collector, event_manager, agent):
        """Test VLM inference completed handler records metrics."""

        event = VLMInferenceCompletedEvent(
            plugin_name="moondream",
            model="moondream-cloud",
            text="A person walking",
            latency_ms=200.0,
            frames_processed=5,
            input_tokens=100,
            output_tokens=20,
        )
        event_manager.send(event)
        await event_manager.wait(1)

        vlm_inferences.add.assert_called_once_with(
            1, {"provider": "moondream", "model": "moondream-cloud"}
        )
        vlm_inference_latency_ms.record.assert_called_once_with(
            200.0, {"provider": "moondream", "model": "moondream-cloud"}
        )
        video_frames_processed.add.assert_called_once_with(
            5, {"provider": "moondream", "model": "moondream-cloud"}
        )
        vlm_input_tokens.add.assert_called_once_with(
            100, {"provider": "moondream", "model": "moondream-cloud"}
        )
        vlm_output_tokens.add.assert_called_once_with(
            20, {"provider": "moondream", "model": "moondream-cloud"}
        )

        assert agent.metrics.vlm_inferences__total.value() == 1
        assert agent.metrics.vlm_inference_latency_ms__avg.value() == 200.0
        assert agent.metrics.video_frames_processed__total.value() == 5
        assert agent.metrics.vlm_input_tokens__total.value() == 100
        assert agent.metrics.vlm_output_tokens__total.value() == 20

    async def test_on_vlm_error(self, collector, event_manager, agent):
        """Test VLM error handler records metrics."""

        event = VLMErrorEvent(
            plugin_name="moondream",
            error=RuntimeError("Inference failed"),
            error_code="INFERENCE_ERROR",
        )
        event_manager.send(event)
        await event_manager.wait(1)

        vlm_errors.add.assert_called_once_with(
            1,
            {
                "provider": "moondream",
                "error_type": "RuntimeError",
                "error_code": "INFERENCE_ERROR",
            },
        )

    async def test_base_attributes_extracts_provider(self, collector):
        """Test that base attributes correctly extracts provider."""

        event = LLMResponseCompletedEvent(
            plugin_name="test_provider",
            text="Hello",
        )

        attrs = collector._base_attributes(event)
        assert attrs == {"provider": "test_provider"}

    async def test_base_attributes_handles_missing_plugin_name(self, collector):
        """Test that base attributes handles missing plugin_name."""

        event = LLMResponseCompletedEvent(
            text="Hello",
        )
        attrs = collector._base_attributes(event)
        assert attrs == {}
