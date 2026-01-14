"""MetricsCollector that subscribes to events and records OpenTelemetry metrics.

This class bridges the event system with OpenTelemetry metrics. It subscribes to
events from LLM, STT, TTS, turn detection, and realtime components and records the
corresponding metrics.

Usage:
    from vision_agents.core.observability import MetricsCollector

    agent = Agent(llm=OpenAILLM(), stt=DeepgramSTT(), tts=CartesiaTTS())
    collector = MetricsCollector(agent)

    # Metrics are now automatically recorded when events are emitted
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Dict

# Import event types at module level so they can be resolved by typing.get_type_hints()
from vision_agents.core.events import PluginBaseEvent, VideoProcessorDetectionEvent
from vision_agents.core.llm.events import (
    LLMErrorEvent,
    LLMResponseCompletedEvent,
    RealtimeAgentSpeechTranscriptionEvent,
    RealtimeAudioInputEvent,
    RealtimeAudioOutputEvent,
    RealtimeConnectedEvent,
    RealtimeDisconnectedEvent,
    RealtimeErrorEvent,
    RealtimeResponseEvent,
    RealtimeUserSpeechTranscriptionEvent,
    ToolEndEvent,
    VLMErrorEvent,
    VLMInferenceCompletedEvent,
)
from vision_agents.core.stt.events import STTErrorEvent, STTTranscriptEvent
from vision_agents.core.tts.events import TTSErrorEvent, TTSSynthesisCompleteEvent
from vision_agents.core.turn_detection.events import TurnEndedEvent

from . import metrics

if TYPE_CHECKING:
    from vision_agents.core.agents import Agent

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects metrics from agent events and records them to OpenTelemetry.

    This class subscribes to events from the agent's LLM, STT, TTS, turn
    detection, and realtime components. When events are emitted, it extracts
    relevant data and records OpenTelemetry metrics.

    Attributes:
        agent: The agent to collect metrics from.
    """

    def __init__(self, agent: Agent):
        """Initialize the metrics collector.

        Args:
            agent: The agent to collect metrics from.
        """
        self.agent = agent
        self._agent_metrics = agent.metrics
        # Track realtime session start times for duration calculation
        self._realtime_session_starts: Dict[str, float] = {}
        self._subscribe()

    def _subscribe(self) -> None:
        """Subscribe to all relevant events from the agent's components."""
        self._subscribe_to_llm_events()
        self._subscribe_to_realtime_events()
        self._subscribe_to_stt_events()
        self._subscribe_to_tts_events()
        self._subscribe_to_turn_detection_events()
        self._subscribe_to_vlm_events()
        self._subscribe_to_processor_events()

    def _subscribe_to_llm_events(self) -> None:
        """Subscribe to LLM events."""
        if not self.agent.llm:
            return

        @self.agent.llm.events.subscribe
        async def on_llm_response_completed(event: LLMResponseCompletedEvent):
            self._on_llm_response_completed(event)

        @self.agent.llm.events.subscribe
        async def on_tool_end(event: ToolEndEvent):
            self._on_tool_end(event)

        @self.agent.llm.events.subscribe
        async def on_llm_error(event: LLMErrorEvent):
            self._on_llm_error(event)

    def _subscribe_to_realtime_events(self) -> None:
        """Subscribe to Realtime LLM events."""
        if not self.agent.llm:
            return

        @self.agent.llm.events.subscribe
        async def on_realtime_connected(event: RealtimeConnectedEvent):
            self._on_realtime_connected(event)

        @self.agent.llm.events.subscribe
        async def on_realtime_disconnected(event: RealtimeDisconnectedEvent):
            self._on_realtime_disconnected(event)

        @self.agent.llm.events.subscribe
        async def on_realtime_audio_input(event: RealtimeAudioInputEvent):
            self._on_realtime_audio_input(event)

        @self.agent.llm.events.subscribe
        async def on_realtime_audio_output(event: RealtimeAudioOutputEvent):
            self._on_realtime_audio_output(event)

        @self.agent.llm.events.subscribe
        async def on_realtime_response(event: RealtimeResponseEvent):
            self._on_realtime_response(event)

        @self.agent.llm.events.subscribe
        async def on_realtime_user_transcription(
            event: RealtimeUserSpeechTranscriptionEvent,
        ):
            self._on_realtime_user_transcription(event)

        @self.agent.llm.events.subscribe
        async def on_realtime_agent_transcription(
            event: RealtimeAgentSpeechTranscriptionEvent,
        ):
            self._on_realtime_agent_transcription(event)

        @self.agent.llm.events.subscribe
        async def on_realtime_error(event: RealtimeErrorEvent):
            self._on_realtime_error(event)

    def _subscribe_to_stt_events(self) -> None:
        """Subscribe to STT events."""
        if not self.agent.stt:
            return

        @self.agent.stt.events.subscribe
        async def on_stt_transcript(event: STTTranscriptEvent):
            self._on_stt_transcript(event)

        @self.agent.stt.events.subscribe
        async def on_stt_error(event: STTErrorEvent):
            self._on_stt_error(event)

    def _subscribe_to_tts_events(self) -> None:
        """Subscribe to TTS events."""
        if not self.agent.tts:
            return

        @self.agent.tts.events.subscribe
        async def on_tts_synthesis_complete(event: TTSSynthesisCompleteEvent):
            self._on_tts_synthesis_complete(event)

        @self.agent.tts.events.subscribe
        async def on_tts_error(event: TTSErrorEvent):
            self._on_tts_error(event)

    def _subscribe_to_turn_detection_events(self) -> None:
        """Subscribe to turn detection events."""
        if not self.agent.turn_detection:
            return

        @self.agent.turn_detection.events.subscribe
        async def on_turn_ended(event: TurnEndedEvent):
            self._on_turn_ended(event)

    # =========================================================================
    # LLM Event Handlers
    # =========================================================================

    def _on_llm_response_completed(self, event: LLMResponseCompletedEvent) -> None:
        """Handle LLM response completed event."""
        attrs = self._base_attributes(event)
        if event.model:
            attrs["model"] = event.model

        # Record latency
        if event.latency_ms is not None:
            metrics.llm_latency_ms.record(event.latency_ms, attrs)
            self._agent_metrics.llm_latency_ms__avg.update(event.latency_ms)

        # Record time to first token
        if event.time_to_first_token_ms is not None:
            metrics.llm_time_to_first_token_ms.record(
                event.time_to_first_token_ms, attrs
            )
            self._agent_metrics.llm_time_to_first_token_ms__avg.update(
                event.time_to_first_token_ms
            )

        # Record token usage
        if event.input_tokens is not None:
            metrics.llm_input_tokens.add(event.input_tokens, attrs)
            self._agent_metrics.llm_input_tokens__total.inc(event.input_tokens)
        if event.output_tokens is not None:
            metrics.llm_output_tokens.add(event.output_tokens, attrs)
            self._agent_metrics.llm_output_tokens__total.inc(event.output_tokens)

    def _on_tool_end(self, event: ToolEndEvent) -> None:
        """Handle tool execution end event."""
        attrs = self._base_attributes(event)
        attrs["tool_name"] = event.tool_name
        attrs["success"] = str(event.success).lower()

        metrics.llm_tool_calls.add(1, attrs)
        self._agent_metrics.llm_tool_calls__total.inc(1)

        if event.execution_time_ms is not None:
            metrics.llm_tool_latency_ms.record(event.execution_time_ms, attrs)
            self._agent_metrics.llm_tool_latency_ms__avg.update(event.execution_time_ms)

    def _on_llm_error(self, event: LLMErrorEvent) -> None:
        """Handle LLM error event."""
        attrs = self._base_attributes(event)
        if event.error:
            attrs["error_type"] = type(event.error).__name__
        if event.error_code:
            attrs["error_code"] = event.error_code

        metrics.llm_errors.add(1, attrs)

    # =========================================================================
    # Realtime LLM Event Handlers
    # =========================================================================

    def _on_realtime_connected(self, event: "RealtimeConnectedEvent") -> None:
        """Handle realtime connected event."""
        attrs = self._base_attributes(event)

        # Track session start time
        # TODO: Some lru structure here? this dict will grow if the sessions are not closed properly
        if event.session_id:
            self._realtime_session_starts[event.session_id] = time.perf_counter()

        metrics.realtime_sessions.add(1, attrs)

    def _on_realtime_disconnected(self, event: "RealtimeDisconnectedEvent") -> None:
        """Handle realtime disconnected event."""
        attrs = self._base_attributes(event)
        attrs["was_clean"] = str(event.was_clean).lower()

        # Calculate session duration
        if event.session_id and event.session_id in self._realtime_session_starts:
            start_time = self._realtime_session_starts.pop(event.session_id)
            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics.realtime_session_duration_ms.record(duration_ms, attrs)

    def _on_realtime_audio_input(self, event: "RealtimeAudioInputEvent") -> None:
        """Handle realtime audio input event."""
        attrs = self._base_attributes(event)

        if event.data and event.data.samples is not None:
            # Record bytes using nbytes to handle all audio formats
            metrics.realtime_audio_input_bytes.add(event.data.samples.nbytes, attrs)
            self._agent_metrics.realtime_audio_input_bytes__total.inc(
                event.data.samples.nbytes
            )

            # Record duration
            if event.data.duration_ms:
                metrics.realtime_audio_input_duration_ms.add(
                    int(event.data.duration_ms), attrs
                )
                self._agent_metrics.realtime_audio_input_duration_ms__total.inc(
                    int(event.data.duration_ms)
                )

    def _on_realtime_audio_output(self, event: "RealtimeAudioOutputEvent") -> None:
        """Handle realtime audio output event."""
        attrs = self._base_attributes(event)

        if event.data and event.data.samples is not None:
            # Record bytes using nbytes to handle all audio formats
            metrics.realtime_audio_output_bytes.add(event.data.samples.nbytes, attrs)
            self._agent_metrics.realtime_audio_output_bytes__total.inc(
                event.data.samples.nbytes
            )

            # Record duration
            if event.data.duration_ms:
                metrics.realtime_audio_output_duration_ms.add(
                    int(event.data.duration_ms), attrs
                )
                self._agent_metrics.realtime_audio_output_duration_ms__total.inc(
                    int(event.data.duration_ms)
                )

    def _on_realtime_response(self, event: "RealtimeResponseEvent") -> None:
        """Handle realtime response event."""
        attrs = self._base_attributes(event)
        attrs["is_complete"] = str(event.is_complete).lower()

        if event.is_complete:
            metrics.realtime_responses.add(1, attrs)

    def _on_realtime_user_transcription(
        self, event: "RealtimeUserSpeechTranscriptionEvent"
    ) -> None:
        """Handle realtime user speech transcription event."""
        attrs = self._base_attributes(event)
        metrics.realtime_user_transcriptions.add(1, attrs)
        self._agent_metrics.realtime_user_transcriptions__total.inc(1)

    def _on_realtime_agent_transcription(
        self, event: "RealtimeAgentSpeechTranscriptionEvent"
    ) -> None:
        """Handle realtime agent speech transcription event."""
        attrs = self._base_attributes(event)
        metrics.realtime_agent_transcriptions.add(1, attrs)
        self._agent_metrics.realtime_agent_transcriptions__total.inc(1)

    def _on_realtime_error(self, event: "RealtimeErrorEvent") -> None:
        """Handle realtime error event."""
        attrs = self._base_attributes(event)
        if event.error:
            attrs["error_type"] = type(event.error).__name__
        if event.error_code:
            attrs["error_code"] = event.error_code
        attrs["is_recoverable"] = str(event.is_recoverable).lower()

        metrics.realtime_errors.add(1, attrs)

    # =========================================================================
    # STT Event Handlers
    # =========================================================================

    def _on_stt_transcript(self, event: STTTranscriptEvent) -> None:
        """Handle STT transcript event."""
        attrs = self._base_attributes(event)
        if event.model_name:
            attrs["model"] = event.model_name
        if event.language:
            attrs["language"] = event.language

        if event.processing_time_ms is not None:
            metrics.stt_latency_ms.record(event.processing_time_ms, attrs)
            self._agent_metrics.stt_latency_ms__avg.update(event.processing_time_ms)

        if event.audio_duration_ms is not None:
            metrics.stt_audio_duration_ms.record(event.audio_duration_ms, attrs)
            self._agent_metrics.stt_audio_duration_ms__total.inc(
                int(event.audio_duration_ms)
            )

    def _on_stt_error(self, event: STTErrorEvent) -> None:
        """Handle STT error event."""
        attrs = self._base_attributes(event)
        if event.error:
            attrs["error_type"] = type(event.error).__name__
        if event.error_code:
            attrs["error_code"] = event.error_code

        metrics.stt_errors.add(1, attrs)

    # =========================================================================
    # TTS Event Handlers
    # =========================================================================

    def _on_tts_synthesis_complete(self, event: TTSSynthesisCompleteEvent) -> None:
        """Handle TTS synthesis complete event."""
        attrs = self._base_attributes(event)

        # Record synthesis latency
        if event.synthesis_time_ms is not None:
            metrics.tts_latency_ms.record(event.synthesis_time_ms, attrs)
            self._agent_metrics.tts_latency_ms__avg.update(event.synthesis_time_ms)

        # Record audio duration
        if event.audio_duration_ms is not None:
            metrics.tts_audio_duration_ms.record(event.audio_duration_ms, attrs)
            self._agent_metrics.tts_audio_duration_ms__total.inc(
                int(event.audio_duration_ms)
            )

        # Record characters synthesized
        if event.text:
            metrics.tts_characters.add(len(event.text), attrs)
            self._agent_metrics.tts_characters__total.inc(len(event.text))

    def _on_tts_error(self, event: TTSErrorEvent) -> None:
        """Handle TTS error event."""
        attrs = self._base_attributes(event)
        if event.error:
            attrs["error_type"] = type(event.error).__name__
        if event.error_code:
            attrs["error_code"] = event.error_code

        metrics.tts_errors.add(1, attrs)

    # =========================================================================
    # Turn Detection Event Handlers
    # =========================================================================

    def _on_turn_ended(self, event: TurnEndedEvent) -> None:
        """Handle turn ended event."""
        attrs = self._base_attributes(event)

        if event.duration_ms is not None:
            metrics.turn_duration_ms.record(event.duration_ms, attrs)
            self._agent_metrics.turn_duration_ms__avg.update(event.duration_ms)

        if event.trailing_silence_ms is not None:
            metrics.turn_trailing_silence_ms.record(event.trailing_silence_ms, attrs)
            self._agent_metrics.turn_trailing_silence_ms__avg.update(
                event.trailing_silence_ms
            )

    # =========================================================================
    # VLM Event Handlers
    # =========================================================================

    def _subscribe_to_vlm_events(self) -> None:
        """Subscribe to VLM events from VideoLLM instances."""
        if not self.agent.llm:
            return

        @self.agent.llm.events.subscribe
        async def on_vlm_inference_completed(event: VLMInferenceCompletedEvent):
            self._on_vlm_inference_completed(event)

        @self.agent.llm.events.subscribe
        async def on_vlm_error(event: VLMErrorEvent):
            self._on_vlm_error(event)

    def _on_vlm_inference_completed(self, event: VLMInferenceCompletedEvent) -> None:
        """Handle VLM inference completed event."""
        attrs = self._base_attributes(event)
        if event.model:
            attrs["model"] = event.model

        # Record inference count
        metrics.vlm_inferences.add(1, attrs)
        self._agent_metrics.vlm_inferences__total.inc(1)

        # Record latency
        if event.latency_ms is not None:
            metrics.vlm_inference_latency_ms.record(event.latency_ms, attrs)
            self._agent_metrics.vlm_inference_latency_ms__avg.update(event.latency_ms)

        # Record token usage
        if event.input_tokens is not None:
            metrics.vlm_input_tokens.add(event.input_tokens, attrs)
            self._agent_metrics.vlm_input_tokens__total.inc(event.input_tokens)
        if event.output_tokens is not None:
            metrics.vlm_output_tokens.add(event.output_tokens, attrs)
            self._agent_metrics.vlm_output_tokens__total.inc(event.output_tokens)

        # Record video-specific metrics
        if event.frames_processed > 0:
            metrics.video_frames_processed.add(event.frames_processed, attrs)
            self._agent_metrics.video_frames_processed__total.inc(
                event.frames_processed
            )
        if event.detections > 0:
            metrics.video_detections.add(event.detections, attrs)

    def _on_vlm_error(self, event: VLMErrorEvent) -> None:
        """Handle VLM error event."""
        attrs = self._base_attributes(event)
        if event.error:
            attrs["error_type"] = type(event.error).__name__
        if event.error_code:
            attrs["error_code"] = event.error_code

        metrics.vlm_errors.add(1, attrs)

    # =========================================================================
    # Video Processor Event Handlers
    # =========================================================================

    def _subscribe_to_processor_events(self) -> None:
        """Subscribe to video processor events from any registered processors."""

        # Subscribe to agent-level events for processor events
        # Processors emit events through the agent's event system
        @self.agent.events.subscribe
        async def on_detection_completed(event: VideoProcessorDetectionEvent):
            self._on_detection_completed(event)

    def _on_detection_completed(self, event: VideoProcessorDetectionEvent) -> None:
        """Handle video detection completed event."""
        attrs = self._base_attributes(event)

        # Add model info if available
        if event.model_id:
            attrs["model"] = event.model_id

        # Record detection count
        if event.detection_count > 0:
            metrics.video_detections.add(event.detection_count, attrs)

        # Record frame processed
        metrics.video_frames_processed.add(1, attrs)
        self._agent_metrics.video_frames_processed__total.inc(1)

        # Record inference latency if available
        if event.inference_time_ms is not None:
            metrics.video_processing_latency_ms.record(event.inference_time_ms, attrs)
            self._agent_metrics.video_processing_latency_ms__avg.update(
                event.inference_time_ms
            )

    # =========================================================================
    # Helpers
    # =========================================================================

    def _base_attributes(self, event: PluginBaseEvent) -> dict:
        """Extract base attributes from an event.

        Args:
            event: The event to extract attributes from.

        Returns:
            Dictionary of base attributes.
        """
        attrs = {}
        if event.plugin_name:
            attrs["provider"] = event.plugin_name
        return attrs
