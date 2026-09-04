import logging
from typing import Self

from . import metrics
from .agent import AgentMetrics

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects metrics into ``AgentMetrics`` and OpenTelemetry meters.

    Callers invoke ``on_*`` methods directly; each method updates the local
    ``AgentMetrics`` and, at the root collector, writes to OpenTelemetry.
    Child collectors forward to their parent, so OTel is emitted exactly
    once per call regardless of merge depth.
    """

    def __init__(self):
        self.agent_metrics = AgentMetrics()
        self.parent: Self | None = None

    def merge(self, child: Self) -> None:
        """Make this collector the parent of ``child``.

        After merge, every metric handler invoked on ``child`` updates
        ``child``'s own ``AgentMetrics`` and is then forwarded to this
        collector (and onward up the chain). Only the root collector
        (``parent is None``) writes to OpenTelemetry meters, keeping OTel
        writes singular per event.

        Re-merging the same child into the same parent is a no-op.
        Merging a child that already has a different parent reparents it
        to this collector.

        Raises:
            ValueError: if ``child is self`` or the merge would create a
                cycle.
        """
        if child is self:
            raise ValueError("cannot merge a collector into itself")
        if child.parent is self:
            return
        node: Self | None = self
        while node is not None:
            if node is child:
                raise ValueError("merge would create a cycle")
            node = node.parent
        child.parent = self

    # =========================================================================
    # LLM
    # =========================================================================

    def on_llm_response(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        latency_ms: float | None = None,
        time_to_first_token_ms: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        """Record a completed LLM response."""
        if latency_ms is not None:
            self.agent_metrics.llm_latency_ms__avg.update(latency_ms)
        if time_to_first_token_ms is not None:
            self.agent_metrics.llm_time_to_first_token_ms__avg.update(
                time_to_first_token_ms
            )
        if input_tokens is not None:
            self.agent_metrics.llm_input_tokens__total.inc(input_tokens)
        if output_tokens is not None:
            self.agent_metrics.llm_output_tokens__total.inc(output_tokens)

        if self.parent is not None:
            self.parent.on_llm_response(
                provider=provider,
                model=model,
                latency_ms=latency_ms,
                time_to_first_token_ms=time_to_first_token_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            return

        attrs: dict = {}
        if provider:
            attrs["provider"] = provider
        if model:
            attrs["model"] = model
        if latency_ms is not None:
            metrics.llm_latency_ms.record(latency_ms, attrs)
        if time_to_first_token_ms is not None:
            metrics.llm_time_to_first_token_ms.record(time_to_first_token_ms, attrs)
        if input_tokens is not None:
            metrics.llm_input_tokens.add(input_tokens, attrs)
        if output_tokens is not None:
            metrics.llm_output_tokens.add(output_tokens, attrs)

    def on_tool_call(
        self,
        *,
        tool_name: str,
        success: bool,
        provider: str | None = None,
        execution_time_ms: float | None = None,
    ) -> None:
        """Record a tool/function call executed by the LLM."""
        self.agent_metrics.llm_tool_calls__total.inc(1)
        if execution_time_ms is not None:
            self.agent_metrics.llm_tool_latency_ms__avg.update(execution_time_ms)

        if self.parent is not None:
            self.parent.on_tool_call(
                tool_name=tool_name,
                success=success,
                provider=provider,
                execution_time_ms=execution_time_ms,
            )
            return

        attrs: dict = {"tool_name": tool_name, "success": str(success).lower()}
        if provider:
            attrs["provider"] = provider
        metrics.llm_tool_calls.add(1, attrs)
        if execution_time_ms is not None:
            metrics.llm_tool_latency_ms.record(execution_time_ms, attrs)

    def on_llm_error(
        self,
        *,
        provider: str | None = None,
        error_type: str | None = None,
        error_code: str | None = None,
    ) -> None:
        """Record an LLM error."""
        if self.parent is not None:
            self.parent.on_llm_error(
                provider=provider, error_type=error_type, error_code=error_code
            )
            return

        attrs: dict = {}
        if provider:
            attrs["provider"] = provider
        if error_type:
            attrs["error_type"] = error_type
        if error_code:
            attrs["error_code"] = error_code
        metrics.llm_errors.add(1, attrs)

    # =========================================================================
    # Realtime LLM
    # =========================================================================

    def on_realtime_audio_input(
        self,
        *,
        byte_count: int,
        provider: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """Record audio sent to a realtime LLM."""
        self.agent_metrics.realtime_audio_input_bytes__total.inc(byte_count)
        if duration_ms is not None:
            self.agent_metrics.realtime_audio_input_duration_ms__total.inc(
                int(duration_ms)
            )

        if self.parent is not None:
            self.parent.on_realtime_audio_input(
                byte_count=byte_count, provider=provider, duration_ms=duration_ms
            )
            return

        attrs: dict = {}
        if provider:
            attrs["provider"] = provider
        metrics.realtime_audio_input_bytes.add(byte_count, attrs)
        if duration_ms is not None:
            metrics.realtime_audio_input_duration_ms.add(int(duration_ms), attrs)

    def on_realtime_audio_output(
        self,
        *,
        byte_count: int,
        provider: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """Record audio received from a realtime LLM."""
        self.agent_metrics.realtime_audio_output_bytes__total.inc(byte_count)
        if duration_ms is not None:
            self.agent_metrics.realtime_audio_output_duration_ms__total.inc(
                int(duration_ms)
            )

        if self.parent is not None:
            self.parent.on_realtime_audio_output(
                byte_count=byte_count, provider=provider, duration_ms=duration_ms
            )
            return

        attrs: dict = {}
        if provider:
            attrs["provider"] = provider
        metrics.realtime_audio_output_bytes.add(byte_count, attrs)
        if duration_ms is not None:
            metrics.realtime_audio_output_duration_ms.add(int(duration_ms), attrs)

    def on_realtime_response_completed(self, *, provider: str | None = None) -> None:
        """Record a completed realtime response."""
        if self.parent is not None:
            self.parent.on_realtime_response_completed(provider=provider)
            return

        attrs: dict = {}
        if provider:
            attrs["provider"] = provider
        metrics.realtime_responses.add(1, attrs)

    def on_realtime_user_transcription(self, *, provider: str | None = None) -> None:
        """Record a user speech transcription from a realtime LLM."""
        self.agent_metrics.realtime_user_transcriptions__total.inc(1)

        if self.parent is not None:
            self.parent.on_realtime_user_transcription(provider=provider)
            return

        attrs: dict = {}
        if provider:
            attrs["provider"] = provider
        metrics.realtime_user_transcriptions.add(1, attrs)

    def on_realtime_agent_transcription(self, *, provider: str | None = None) -> None:
        """Record an agent speech transcription from a realtime LLM."""
        self.agent_metrics.realtime_agent_transcriptions__total.inc(1)

        if self.parent is not None:
            self.parent.on_realtime_agent_transcription(provider=provider)
            return

        attrs: dict = {}
        if provider:
            attrs["provider"] = provider
        metrics.realtime_agent_transcriptions.add(1, attrs)

    def on_realtime_error(
        self,
        *,
        provider: str | None = None,
        error_type: str | None = None,
        error_code: str | None = None,
        is_recoverable: bool = False,
    ) -> None:
        """Record a realtime LLM error."""
        if self.parent is not None:
            self.parent.on_realtime_error(
                provider=provider,
                error_type=error_type,
                error_code=error_code,
                is_recoverable=is_recoverable,
            )
            return

        attrs: dict = {"is_recoverable": str(is_recoverable).lower()}
        if provider:
            attrs["provider"] = provider
        if error_type:
            attrs["error_type"] = error_type
        if error_code:
            attrs["error_code"] = error_code
        metrics.realtime_errors.add(1, attrs)

    # =========================================================================
    # STT
    # =========================================================================

    def on_stt_transcript(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        language: str | None = None,
        processing_time_ms: float | None = None,
        audio_duration_ms: float | None = None,
    ) -> None:
        """Record a completed STT transcript."""
        if processing_time_ms is not None:
            self.agent_metrics.stt_latency_ms__avg.update(processing_time_ms)
        if audio_duration_ms is not None:
            self.agent_metrics.stt_audio_duration_ms__total.inc(int(audio_duration_ms))

        if self.parent is not None:
            self.parent.on_stt_transcript(
                provider=provider,
                model=model,
                language=language,
                processing_time_ms=processing_time_ms,
                audio_duration_ms=audio_duration_ms,
            )
            return

        attrs: dict = {}
        if provider:
            attrs["provider"] = provider
        if model:
            attrs["model"] = model
        if language:
            attrs["language"] = language
        if processing_time_ms is not None:
            metrics.stt_latency_ms.record(processing_time_ms, attrs)
        if audio_duration_ms is not None:
            metrics.stt_audio_duration_ms.record(audio_duration_ms, attrs)

    def on_stt_error(
        self,
        *,
        provider: str | None = None,
        error_type: str | None = None,
        error_code: str | None = None,
    ) -> None:
        """Record an STT error."""
        if self.parent is not None:
            self.parent.on_stt_error(
                provider=provider, error_type=error_type, error_code=error_code
            )
            return

        attrs: dict = {}
        if provider:
            attrs["provider"] = provider
        if error_type:
            attrs["error_type"] = error_type
        if error_code:
            attrs["error_code"] = error_code
        metrics.stt_errors.add(1, attrs)

    # =========================================================================
    # TTS
    # =========================================================================

    def on_tts_synthesis(
        self,
        *,
        provider: str | None = None,
        synthesis_time_ms: float | None = None,
        audio_duration_ms: float | None = None,
        character_count: int | None = None,
    ) -> None:
        """Record a completed TTS synthesis."""
        if synthesis_time_ms is not None:
            self.agent_metrics.tts_latency_ms__avg.update(synthesis_time_ms)
        if audio_duration_ms is not None:
            self.agent_metrics.tts_audio_duration_ms__total.inc(int(audio_duration_ms))
        if character_count is not None:
            self.agent_metrics.tts_characters__total.inc(character_count)

        if self.parent is not None:
            self.parent.on_tts_synthesis(
                provider=provider,
                synthesis_time_ms=synthesis_time_ms,
                audio_duration_ms=audio_duration_ms,
                character_count=character_count,
            )
            return

        attrs: dict = {}
        if provider:
            attrs["provider"] = provider
        if synthesis_time_ms is not None:
            metrics.tts_latency_ms.record(synthesis_time_ms, attrs)
        if audio_duration_ms is not None:
            metrics.tts_audio_duration_ms.record(audio_duration_ms, attrs)
        if character_count is not None:
            metrics.tts_characters.add(character_count, attrs)

    def on_tts_error(
        self,
        *,
        provider: str | None = None,
        error_type: str | None = None,
        error_code: str | None = None,
    ) -> None:
        """Record a TTS error."""
        if self.parent is not None:
            self.parent.on_tts_error(
                provider=provider, error_type=error_type, error_code=error_code
            )
            return

        attrs: dict = {}
        if provider:
            attrs["provider"] = provider
        if error_type:
            attrs["error_type"] = error_type
        if error_code:
            attrs["error_code"] = error_code
        metrics.tts_errors.add(1, attrs)

    # =========================================================================
    # Turn Detection
    # =========================================================================

    def on_turn_ended(
        self,
        *,
        provider: str | None = None,
        duration_ms: float | None = None,
        trailing_silence_ms: float | None = None,
    ) -> None:
        """Record a completed conversational turn."""
        if duration_ms is not None:
            self.agent_metrics.turn_duration_ms__avg.update(duration_ms)
        if trailing_silence_ms is not None:
            self.agent_metrics.turn_trailing_silence_ms__avg.update(trailing_silence_ms)

        if self.parent is not None:
            self.parent.on_turn_ended(
                provider=provider,
                duration_ms=duration_ms,
                trailing_silence_ms=trailing_silence_ms,
            )
            return

        attrs: dict = {}
        if provider:
            attrs["provider"] = provider
        if duration_ms is not None:
            metrics.turn_duration_ms.record(duration_ms, attrs)
        if trailing_silence_ms is not None:
            metrics.turn_trailing_silence_ms.record(trailing_silence_ms, attrs)

    # =========================================================================
    # VLM
    # =========================================================================

    def on_vlm_inference(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        latency_ms: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        frames_processed: int = 0,
        detections: int = 0,
    ) -> None:
        """Record a completed VLM inference."""
        self.agent_metrics.vlm_inferences__total.inc(1)
        if latency_ms is not None:
            self.agent_metrics.vlm_inference_latency_ms__avg.update(latency_ms)
        if input_tokens is not None:
            self.agent_metrics.vlm_input_tokens__total.inc(input_tokens)
        if output_tokens is not None:
            self.agent_metrics.vlm_output_tokens__total.inc(output_tokens)
        if frames_processed > 0:
            self.agent_metrics.video_frames_processed__total.inc(frames_processed)

        if self.parent is not None:
            self.parent.on_vlm_inference(
                provider=provider,
                model=model,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                frames_processed=frames_processed,
                detections=detections,
            )
            return

        attrs: dict = {}
        if provider:
            attrs["provider"] = provider
        if model:
            attrs["model"] = model
        metrics.vlm_inferences.add(1, attrs)
        if latency_ms is not None:
            metrics.vlm_inference_latency_ms.record(latency_ms, attrs)
        if input_tokens is not None:
            metrics.vlm_input_tokens.add(input_tokens, attrs)
        if output_tokens is not None:
            metrics.vlm_output_tokens.add(output_tokens, attrs)
        if frames_processed > 0:
            metrics.video_frames_processed.add(frames_processed, attrs)
        if detections > 0:
            metrics.video_detections.add(detections, attrs)

    def on_vlm_error(
        self,
        *,
        provider: str | None = None,
        error_type: str | None = None,
        error_code: str | None = None,
    ) -> None:
        """Record a VLM error."""
        if self.parent is not None:
            self.parent.on_vlm_error(
                provider=provider, error_type=error_type, error_code=error_code
            )
            return

        attrs: dict = {}
        if provider:
            attrs["provider"] = provider
        if error_type:
            attrs["error_type"] = error_type
        if error_code:
            attrs["error_code"] = error_code
        metrics.vlm_errors.add(1, attrs)

    # =========================================================================
    # Video Processor
    # =========================================================================

    def on_video_detection(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        detection_count: int = 0,
        inference_time_ms: float | None = None,
    ) -> None:
        """Record a video frame processed by a detection processor."""
        self.agent_metrics.video_frames_processed__total.inc(1)
        if inference_time_ms is not None:
            self.agent_metrics.video_processing_latency_ms__avg.update(
                inference_time_ms
            )

        if self.parent is not None:
            self.parent.on_video_detection(
                provider=provider,
                model=model,
                detection_count=detection_count,
                inference_time_ms=inference_time_ms,
            )
            return

        attrs: dict = {}
        if provider:
            attrs["provider"] = provider
        if model:
            attrs["model"] = model
        if detection_count > 0:
            metrics.video_detections.add(detection_count, attrs)
        metrics.video_frames_processed.add(1, attrs)
        if inference_time_ms is not None:
            metrics.video_processing_latency_ms.record(inference_time_ms, attrs)
