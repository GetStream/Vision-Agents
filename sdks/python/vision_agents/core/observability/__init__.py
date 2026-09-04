"""Stream Agents Observability Package.

This package provides observability tools including metrics collection for Stream Agents.

Usage:
    # Configure OpenTelemetry in your application first, then:
    from vision_agents.core.observability import MetricsCollector

    agent = Agent(llm=OpenAILLM(), stt=DeepgramSTT(), tts=CartesiaTTS())
    collector = MetricsCollector(agent)  # Opt-in to metrics collection
"""

from .agent import AgentMetrics
from .collector import MetricsCollector
from .metrics import (
    llm_errors,
    llm_input_tokens,
    # LLM metrics
    llm_latency_ms,
    llm_output_tokens,
    llm_time_to_first_token_ms,
    llm_tool_calls,
    llm_tool_latency_ms,
    meter,
    realtime_agent_transcriptions,
    realtime_audio_input_bytes,
    realtime_audio_input_duration_ms,
    realtime_audio_output_bytes,
    realtime_audio_output_duration_ms,
    realtime_errors,
    realtime_responses,
    realtime_session_duration_ms,
    # Realtime LLM metrics
    realtime_sessions,
    realtime_user_transcriptions,
    stt_audio_duration_ms,
    stt_errors,
    # STT metrics
    stt_latency_ms,
    tts_audio_duration_ms,
    tts_characters,
    tts_errors,
    # TTS metrics
    tts_latency_ms,
    # Turn detection metrics
    turn_duration_ms,
    turn_trailing_silence_ms,
    video_detections,
    # Video processor metrics
    video_frames_processed,
    video_processing_latency_ms,
    vlm_errors,
    # VLM metrics
    vlm_inference_latency_ms,
    vlm_inferences,
    vlm_input_tokens,
    vlm_output_tokens,
)

__all__ = [
    # Main class
    "MetricsCollector",
    # Meter
    "meter",
    # STT metrics
    "stt_latency_ms",
    "stt_audio_duration_ms",
    "stt_errors",
    # TTS metrics
    "tts_latency_ms",
    "tts_audio_duration_ms",
    "tts_characters",
    "tts_errors",
    # LLM metrics
    "llm_latency_ms",
    "llm_time_to_first_token_ms",
    "llm_input_tokens",
    "llm_output_tokens",
    "llm_errors",
    "llm_tool_calls",
    "llm_tool_latency_ms",
    # Turn detection metrics
    "turn_duration_ms",
    "turn_trailing_silence_ms",
    # Realtime LLM metrics
    "realtime_sessions",
    "realtime_session_duration_ms",
    "realtime_audio_input_bytes",
    "realtime_audio_output_bytes",
    "realtime_audio_input_duration_ms",
    "realtime_audio_output_duration_ms",
    "realtime_responses",
    "realtime_user_transcriptions",
    "realtime_agent_transcriptions",
    "realtime_errors",
    # VLM metrics
    "vlm_inference_latency_ms",
    "vlm_inferences",
    "vlm_input_tokens",
    "vlm_output_tokens",
    "vlm_errors",
    # Video processor metrics
    "video_frames_processed",
    "video_processing_latency_ms",
    "video_detections",
    "AgentMetrics",
]
