"""Telnyx plugin for Vision Agents."""

from .audio import (
    TELNYX_DEFAULT_SAMPLE_RATE,
    TELNYX_L16_SAMPLE_RATE,
    l16_to_pcm,
    pcma_to_pcm,
    pcm_to_l16,
    pcm_to_pcma,
    pcm_to_pcmu,
    pcm_to_telnyx_payload,
    pcmu_to_pcm,
    telnyx_payload_to_pcm,
)
from .call_registry import TelnyxCall, TelnyxCallRegistry
from .media_stream import TelnyxMediaFormat, TelnyxMediaStream, attach_phone_to_call

CallRegistry = TelnyxCallRegistry
MediaStream = TelnyxMediaStream

__all__ = [
    "CallRegistry",
    "MediaStream",
    "TELNYX_DEFAULT_SAMPLE_RATE",
    "TELNYX_L16_SAMPLE_RATE",
    "TelnyxCall",
    "TelnyxCallRegistry",
    "TelnyxMediaFormat",
    "TelnyxMediaStream",
    "attach_phone_to_call",
    "l16_to_pcm",
    "pcma_to_pcm",
    "pcm_to_l16",
    "pcm_to_pcma",
    "pcm_to_pcmu",
    "pcm_to_telnyx_payload",
    "pcmu_to_pcm",
    "telnyx_payload_to_pcm",
]
