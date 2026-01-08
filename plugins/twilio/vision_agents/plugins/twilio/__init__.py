"""Twilio plugin for Vision Agents."""

from .audio import mulaw_to_pcm, pcm_to_mulaw, TWILIO_SAMPLE_RATE
from .call_registry import TwilioCall, TwilioCallRegistry
from .media_stream import TwilioMediaStream, attach_phone_to_call
from .models import (
    CallWebhookInput,
    TwilioSignatureVerifier,
    create_media_stream_response,
    create_media_stream_twiml,
    verify_twilio_signature,
)

__all__ = [
    "CallWebhookInput",
    "TwilioCall",
    "TwilioCallRegistry",
    "TwilioMediaStream",
    "TwilioSignatureVerifier",
    "attach_phone_to_call",
    "create_media_stream_response",
    "create_media_stream_twiml",
    "mulaw_to_pcm",
    "pcm_to_mulaw",
    "verify_twilio_signature",
    "TWILIO_SAMPLE_RATE",
]
