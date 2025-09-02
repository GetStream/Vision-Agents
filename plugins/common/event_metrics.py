"""
Event metrics calculation utilities for GetStream AI plugins.

This module provides functions for calculating performance metrics
from events across different plugin types.
"""

from typing import Dict, Any, List

from .events import (
    BaseEvent, STTTranscriptEvent, STTPartialTranscriptEvent,
    TTSAudioEvent, TTSSynthesisStartEvent, TTSSynthesisCompleteEvent,
    VADAudioEvent, VADPartialEvent
)


def calculate_stt_metrics(events: List[BaseEvent]) -> Dict[str, Any]:
    """Calculate STT-specific metrics."""
    transcript_events = [
        e for e in events
        if isinstance(e, (STTTranscriptEvent, STTPartialTranscriptEvent))
    ]

    if not transcript_events:
        return {"total_transcripts": 0}

    # Calculate processing time statistics
    processing_times = [
        e.processing_time_ms for e in transcript_events
        if hasattr(e, 'processing_time_ms') and e.processing_time_ms
    ]

    # Calculate confidence statistics
    confidences = [
        e.confidence for e in transcript_events
        if hasattr(e, 'confidence') and e.confidence is not None
    ]

    metrics = {
        "total_transcripts": len(transcript_events),
        "final_transcripts": len([e for e in transcript_events if getattr(e, 'is_final', True)]),
        "partial_transcripts": len([e for e in transcript_events if not getattr(e, 'is_final', True)]),
    }

    if processing_times:
        metrics.update({
            "avg_processing_time_ms": sum(processing_times) / len(processing_times),
            "min_processing_time_ms": min(processing_times),
            "max_processing_time_ms": max(processing_times),
        })

    if confidences:
        metrics.update({
            "avg_confidence": sum(confidences) / len(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
        })

    return metrics


def calculate_tts_metrics(events: List[BaseEvent]) -> Dict[str, Any]:
    """Calculate TTS-specific metrics."""
    audio_events = [e for e in events if isinstance(e, TTSAudioEvent)]
    synthesis_events = [e for e in events if isinstance(e, TTSSynthesisStartEvent)]
    completion_events = [e for e in events if isinstance(e, TTSSynthesisCompleteEvent)]

    metrics = {
        "total_audio_chunks": len(audio_events),
        "total_syntheses": len(synthesis_events),
        "completed_syntheses": len(completion_events),
    }

    if completion_events:
        synthesis_times = [e.synthesis_time_ms for e in completion_events]
        real_time_factors = [
            e.real_time_factor for e in completion_events
            if e.real_time_factor is not None
        ]

        metrics.update({
            "avg_synthesis_time_ms": sum(synthesis_times) / len(synthesis_times),
            "min_synthesis_time_ms": min(synthesis_times),
            "max_synthesis_time_ms": max(synthesis_times),
        })

        if real_time_factors:
            metrics.update({
                "avg_real_time_factor": sum(real_time_factors) / len(real_time_factors),
                "min_real_time_factor": min(real_time_factors),
                "max_real_time_factor": max(real_time_factors),
            })

    return metrics


def calculate_vad_metrics(events: List[BaseEvent]) -> Dict[str, Any]:
    """Calculate VAD-specific metrics."""
    audio_events = [e for e in events if isinstance(e, VADAudioEvent)]
    partial_events = [e for e in events if isinstance(e, VADPartialEvent)]

    metrics = {
        "total_speech_segments": len(audio_events),
        "total_partial_events": len(partial_events),
    }

    if audio_events:
        durations = [
            e.duration_ms for e in audio_events
            if e.duration_ms is not None
        ]
        probabilities = [
            e.speech_probability for e in audio_events
            if e.speech_probability is not None
        ]

        if durations:
            metrics.update({
                "avg_speech_duration_ms": sum(durations) / len(durations),
                "total_speech_duration_ms": sum(durations),
            })

        if probabilities:
            metrics.update({
                "avg_speech_probability": sum(probabilities) / len(probabilities),
            })

    return metrics
