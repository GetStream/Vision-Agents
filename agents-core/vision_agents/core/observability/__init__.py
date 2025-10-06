"""
Stream Agents Observability Package

This package provides observability tools including metrics and tracing for Stream Agents.
"""

from .metrics import (
    tracer,
    meter,
    stt_latency_ms,
    stt_first_byte_ms,
    stt_bytes_streamed,
    stt_errors,
    tts_latency_ms,
    tts_first_byte_ms,
    tts_bytes_streamed,
    tts_errors,
    inflight_ops,
    CALL_ATTRS,
)

__all__ = [
    "tracer",
    "meter",
    "stt_latency_ms",
    "stt_first_byte_ms",
    "stt_bytes_streamed",
    "stt_errors",
    "tts_latency_ms",
    "tts_first_byte_ms",
    "tts_bytes_streamed",
    "tts_errors",
    "inflight_ops",
    "CALL_ATTRS",
]
