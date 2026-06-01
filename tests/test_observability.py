"""Tests for MetricsCollector public ``on_*`` API."""

import dataclasses
from unittest.mock import patch

import pytest
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


@pytest.fixture()
def mock_metrics():
    """Patch every OTel instrument used by the collector so tests can assert on emits."""
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
def collector(mock_metrics) -> MetricsCollector:
    return MetricsCollector()


class TestMetricsCollector:
    """Tests for the public ``on_*`` API on ``MetricsCollector``."""

    def test_on_llm_response(self, collector):
        collector.on_llm_response(
            provider="openai",
            model="gpt-4",
            latency_ms=150.0,
            time_to_first_token_ms=50.0,
            input_tokens=10,
            output_tokens=5,
        )

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
        assert collector.agent_metrics.llm_latency_ms__avg.value() == 150
        assert collector.agent_metrics.llm_time_to_first_token_ms__avg.value() == 50
        assert collector.agent_metrics.llm_input_tokens__total.value() == 10
        assert collector.agent_metrics.llm_output_tokens__total.value() == 5

    def test_on_llm_response_partial_data(self, collector):
        collector.on_llm_response(provider="openai")

        llm_latency_ms.record.assert_not_called()
        llm_time_to_first_token_ms.record.assert_not_called()
        llm_input_tokens.add.assert_not_called()
        llm_output_tokens.add.assert_not_called()

        assert collector.agent_metrics.llm_latency_ms__avg.value() is None
        assert collector.agent_metrics.llm_time_to_first_token_ms__avg.value() is None
        assert collector.agent_metrics.llm_input_tokens__total.value() == 0
        assert collector.agent_metrics.llm_output_tokens__total.value() == 0

    def test_on_tool_call(self, collector):
        collector.on_tool_call(
            provider="openai",
            tool_name="get_weather",
            success=True,
            execution_time_ms=25.0,
        )

        llm_tool_calls.add.assert_called_once_with(
            1, {"tool_name": "get_weather", "success": "true", "provider": "openai"}
        )
        llm_tool_latency_ms.record.assert_called_once_with(
            25.0, {"tool_name": "get_weather", "success": "true", "provider": "openai"}
        )

        assert collector.agent_metrics.llm_tool_calls__total.value() == 1
        assert collector.agent_metrics.llm_tool_latency_ms__avg.value() == 25

    def test_on_llm_error(self, collector):
        collector.on_llm_error(
            provider="openai",
            error_type="ValueError",
            error_code="BAD_REQUEST",
        )

        llm_errors.add.assert_called_once_with(
            1,
            {
                "provider": "openai",
                "error_type": "ValueError",
                "error_code": "BAD_REQUEST",
            },
        )

    def test_on_stt_transcript(self, collector):
        collector.on_stt_transcript(
            provider="deepgram",
            model="nova-2",
            language="en",
            processing_time_ms=100.0,
            audio_duration_ms=2000.0,
        )

        stt_latency_ms.record.assert_called_once_with(
            100.0, {"provider": "deepgram", "model": "nova-2", "language": "en"}
        )
        stt_audio_duration_ms.record.assert_called_once_with(
            2000.0, {"provider": "deepgram", "model": "nova-2", "language": "en"}
        )
        assert collector.agent_metrics.stt_latency_ms__avg.value() == 100.0
        assert collector.agent_metrics.stt_audio_duration_ms__total.value() == 2000.0

    def test_on_stt_error(self, collector):
        collector.on_stt_error(
            provider="deepgram",
            error_type="ValueError",
            error_code="CONNECTION_ERROR",
        )

        stt_errors.add.assert_called_once_with(
            1,
            {
                "provider": "deepgram",
                "error_type": "ValueError",
                "error_code": "CONNECTION_ERROR",
            },
        )

    def test_on_tts_synthesis(self, collector):
        collector.on_tts_synthesis(
            provider="cartesia",
            synthesis_time_ms=50.0,
            audio_duration_ms=1500.0,
            character_count=len("Hello world"),
        )

        tts_latency_ms.record.assert_called_once_with(50.0, {"provider": "cartesia"})
        tts_audio_duration_ms.record.assert_called_once_with(
            1500.0, {"provider": "cartesia"}
        )
        tts_characters.add.assert_called_once_with(
            len("Hello world"), {"provider": "cartesia"}
        )
        assert collector.agent_metrics.tts_latency_ms__avg.value() == 50.0
        assert collector.agent_metrics.tts_audio_duration_ms__total.value() == 1500.0
        assert collector.agent_metrics.tts_characters__total.value() == len(
            "Hello world"
        )

    def test_on_tts_error(self, collector):
        collector.on_tts_error(
            provider="cartesia",
            error_type="RuntimeError",
            error_code="SYNTHESIS_ERROR",
        )

        tts_errors.add.assert_called_once_with(
            1,
            {
                "provider": "cartesia",
                "error_type": "RuntimeError",
                "error_code": "SYNTHESIS_ERROR",
            },
        )

    def test_on_turn_ended(self, collector):
        collector.on_turn_ended(
            provider="smart_turn",
            duration_ms=3500.0,
            trailing_silence_ms=500.0,
        )

        turn_duration_ms.record.assert_called_once_with(
            3500.0, {"provider": "smart_turn"}
        )
        turn_trailing_silence_ms.record.assert_called_once_with(
            500.0, {"provider": "smart_turn"}
        )
        assert collector.agent_metrics.turn_duration_ms__avg.value() == 3500.0
        assert collector.agent_metrics.turn_trailing_silence_ms__avg.value() == 500.0

    def test_on_vlm_inference(self, collector):
        collector.on_vlm_inference(
            provider="moondream",
            model="moondream-cloud",
            latency_ms=200.0,
            input_tokens=100,
            output_tokens=20,
            frames_processed=5,
        )

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

        assert collector.agent_metrics.vlm_inferences__total.value() == 1
        assert collector.agent_metrics.vlm_inference_latency_ms__avg.value() == 200.0
        assert collector.agent_metrics.video_frames_processed__total.value() == 5
        assert collector.agent_metrics.vlm_input_tokens__total.value() == 100
        assert collector.agent_metrics.vlm_output_tokens__total.value() == 20

    def test_on_vlm_error(self, collector):
        collector.on_vlm_error(
            provider="moondream",
            error_type="RuntimeError",
            error_code="INFERENCE_ERROR",
        )

        vlm_errors.add.assert_called_once_with(
            1,
            {
                "provider": "moondream",
                "error_type": "RuntimeError",
                "error_code": "INFERENCE_ERROR",
            },
        )

    def test_on_video_detection(self, collector):
        collector.on_video_detection(
            provider="roboflow",
            model="yolo-v8",
            detection_count=3,
            inference_time_ms=42.0,
        )

        video_detections.add.assert_called_once_with(
            3, {"provider": "roboflow", "model": "yolo-v8"}
        )
        video_frames_processed.add.assert_called_once_with(
            1, {"provider": "roboflow", "model": "yolo-v8"}
        )
        assert collector.agent_metrics.video_frames_processed__total.value() == 1
        assert collector.agent_metrics.video_processing_latency_ms__avg.value() == 42.0

    def test_on_realtime_audio_input(self, collector):
        collector.on_realtime_audio_input(
            provider="openai", byte_count=1024, duration_ms=20.0
        )

        assert collector.agent_metrics.realtime_audio_input_bytes__total.value() == 1024
        assert (
            collector.agent_metrics.realtime_audio_input_duration_ms__total.value()
            == 20
        )

    def test_on_realtime_user_transcription(self, collector):
        collector.on_realtime_user_transcription(provider="openai")
        assert collector.agent_metrics.realtime_user_transcriptions__total.value() == 1

    def test_hierarchy_emits_otel_only_at_root(self, mock_metrics):
        parent = MetricsCollector()
        child = MetricsCollector()
        parent.merge(child)

        child.on_llm_response(provider="openai", latency_ms=100.0, input_tokens=5)

        # OTel emitted exactly once, at root.
        llm_latency_ms.record.assert_called_once_with(100.0, {"provider": "openai"})
        llm_input_tokens.add.assert_called_once_with(5, {"provider": "openai"})

        # AgentMetrics updated on every collector in the chain.
        assert child.agent_metrics.llm_latency_ms__avg.value() == 100.0
        assert child.agent_metrics.llm_input_tokens__total.value() == 5
        assert parent.agent_metrics.llm_latency_ms__avg.value() == 100.0
        assert parent.agent_metrics.llm_input_tokens__total.value() == 5

    def test_merge_is_idempotent_for_same_parent(self, mock_metrics):
        parent = MetricsCollector()
        child = MetricsCollector()
        parent.merge(child)
        parent.merge(child)

        assert child.parent is parent

        child.on_llm_response(provider="openai", input_tokens=3)

        # Re-merging didn't duplicate the chain: OTel emitted once, parent updated once.
        llm_input_tokens.add.assert_called_once_with(3, {"provider": "openai"})
        assert parent.agent_metrics.llm_input_tokens__total.value() == 3

    def test_merge_reparents_to_new_parent(self):
        parent_a = MetricsCollector()
        parent_b = MetricsCollector()
        child = MetricsCollector()

        parent_a.merge(child)
        parent_b.merge(child)

        assert child.parent is parent_b

        child.on_llm_response(provider="openai", input_tokens=7)

        # Forwarding follows the new parent only.
        assert parent_b.agent_metrics.llm_input_tokens__total.value() == 7
        assert parent_a.agent_metrics.llm_input_tokens__total.value() == 0

    def test_merge_self_raises(self):
        collector = MetricsCollector()
        with pytest.raises(ValueError, match="cannot merge a collector into itself"):
            collector.merge(collector)

    def test_merge_cycle_raises(self):
        a = MetricsCollector()
        b = MetricsCollector()
        a.merge(b)
        with pytest.raises(ValueError, match="merge would create a cycle"):
            b.merge(a)


class TestAgentMetrics:
    def test_to_dict_all_fields_success(self):
        metrics = AgentMetrics()
        metrics_dict = metrics.to_dict()
        all_fields = [f.name for f in dataclasses.fields(AgentMetrics)]
        assert set(all_fields) == set(metrics_dict.keys())

    def test_to_dict_some_fields_success(self):
        metrics = AgentMetrics()
        some_fields = ["realtime_agent_transcriptions__total", "tts_characters__total"]
        metrics_dict = metrics.to_dict(fields=some_fields)
        assert set(some_fields) == set(metrics_dict.keys())

    def test_to_dict_some_fields_missing_fail(self):
        metrics = AgentMetrics()
        with pytest.raises(ValueError, match="Unknown field: unknown_field"):
            metrics.to_dict(fields=["unknown_field"])
