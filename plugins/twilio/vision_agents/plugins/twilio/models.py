"""Twilio webhook models for FastAPI."""

import logging
import os
from typing import Optional

from fastapi import Form, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from twilio.request_validator import RequestValidator
from twilio.twiml.voice_response import VoiceResponse, Connect

from .utils import normalize_websocket_url

logger = logging.getLogger(__name__)


def create_media_stream_twiml(websocket_url: str) -> str:
    """
    Create TwiML that starts a media stream to the given websocket URL.

    Args:
        websocket_url: The websocket URL to stream audio to (e.g. wss://example.com/media/123).
                       https:// URLs are automatically converted to wss://.

    Returns:
        TwiML string.
    """
    websocket_url = normalize_websocket_url(websocket_url)
    response = VoiceResponse()
    connect = Connect()
    connect.stream(url=websocket_url)
    response.append(connect)
    return str(response)


def create_media_stream_response(websocket_url: str) -> Response:
    """
    Create a FastAPI Response with TwiML that starts a media stream.

    Args:
        websocket_url: The websocket URL to stream audio to (e.g. wss://example.com/media/123)

    Returns:
        FastAPI Response with TwiML content.
    """
    return Response(
        content=create_media_stream_twiml(websocket_url), media_type="text/xml"
    )


class TwilioSignatureVerifier:
    """
    Verifies Twilio webhook signatures to ensure requests are authentic.

    Uses the TWILIO_AUTH_TOKEN environment variable to validate
    the X-Twilio-Signature header against the request URL and body.

    Example:
        verifier = TwilioSignatureVerifier()

        @app.post("/twilio/voice")
        async def webhook(
            request: Request,
            _: None = Depends(verifier),
            data: CallWebhookInput = Depends(CallWebhookInput.as_form),
        ):
            # Request is verified, safe to process
            ...
    """

    def __init__(self, auth_token: Optional[str] = None):
        """
        Initialize the verifier.

        Args:
            auth_token: Twilio Auth Token. If not provided, reads from
                       TWILIO_AUTH_TOKEN environment variable.
        """
        self._auth_token = auth_token

    @property
    def auth_token(self) -> str:
        """Get the auth token, falling back to environment variable."""
        token = self._auth_token or os.environ.get("TWILIO_AUTH_TOKEN")
        if not token:
            raise ValueError(
                "TWILIO_AUTH_TOKEN environment variable is required for signature verification"
            )
        return token

    async def __call__(self, request: Request) -> None:
        """
        FastAPI dependency that verifies Twilio signature.

        Raises:
            HTTPException: If signature is missing or invalid.
        """
        from fastapi import HTTPException

        signature = request.headers.get("X-Twilio-Signature")
        if not signature:
            logger.warning("Missing X-Twilio-Signature header")
            raise HTTPException(status_code=403, detail="Missing Twilio signature")

        # Get the full URL (Twilio uses the URL for signature validation)
        url = str(request.url)

        # Get form data as dict
        form = await request.form()
        params = {key: value for key, value in form.items()}

        # Validate the signature
        validator = RequestValidator(self.auth_token)
        if not validator.validate(url, params, signature):
            logger.warning(f"Invalid Twilio signature for {url}")
            raise HTTPException(status_code=403, detail="Invalid Twilio signature")

        logger.debug(f"Twilio signature verified for {url}")


# Default verifier instance using TWILIO_AUTH_TOKEN env var
verify_twilio_signature = TwilioSignatureVerifier()


class CallWebhookInput(BaseModel):
    """
    Twilio voice webhook form data.

    This model represents the form data sent by Twilio when a call is received.
    Use with FastAPI's Form() for automatic parsing.

    Example:
        @app.post("/twilio/voice")
        async def webhook(data: CallWebhookInput = Depends(CallWebhookInput.as_form)):
            print(f"Call from {data.caller} to {data.to}")
    """

    # Call identification
    call_sid: str = Field(
        alias="CallSid", description="Unique identifier for this call"
    )
    account_sid: str = Field(alias="AccountSid", description="Twilio account SID")
    api_version: str = Field(alias="ApiVersion", default="2010-04-01")

    # Call status
    call_status: str = Field(
        alias="CallStatus",
        description="Current call status (ringing, in-progress, etc.)",
    )
    direction: str = Field(
        alias="Direction", description="Call direction (inbound or outbound)"
    )

    # From (caller) information
    from_number: str = Field(
        alias="From", description="Caller's phone number (E.164 format)"
    )
    caller: str = Field(
        alias="Caller", description="Caller's phone number (same as From)"
    )
    caller_city: Optional[str] = Field(alias="CallerCity", default=None)
    caller_state: Optional[str] = Field(alias="CallerState", default=None)
    caller_zip: Optional[str] = Field(alias="CallerZip", default=None)
    caller_country: Optional[str] = Field(alias="CallerCountry", default=None)
    from_city: Optional[str] = Field(alias="FromCity", default=None)
    from_state: Optional[str] = Field(alias="FromState", default=None)
    from_zip: Optional[str] = Field(alias="FromZip", default=None)
    from_country: Optional[str] = Field(alias="FromCountry", default=None)

    # To (called) information
    to: str = Field(alias="To", description="Called phone number (E.164 format)")
    called: str = Field(alias="Called", description="Called phone number (same as To)")
    called_city: Optional[str] = Field(alias="CalledCity", default=None)
    called_state: Optional[str] = Field(alias="CalledState", default=None)
    called_zip: Optional[str] = Field(alias="CalledZip", default=None)
    called_country: Optional[str] = Field(alias="CalledCountry", default=None)
    to_city: Optional[str] = Field(alias="ToCity", default=None)
    to_state: Optional[str] = Field(alias="ToState", default=None)
    to_zip: Optional[str] = Field(alias="ToZip", default=None)
    to_country: Optional[str] = Field(alias="ToCountry", default=None)

    # STIR/SHAKEN verification
    stir_verstat: Optional[str] = Field(
        alias="StirVerstat", default=None, description="STIR/SHAKEN verification status"
    )

    # Call token for additional security
    call_token: Optional[str] = Field(alias="CallToken", default=None)

    model_config = {"populate_by_name": True}

    @classmethod
    def as_form(
        cls,
        CallSid: str = Form(""),
        AccountSid: str = Form(""),
        ApiVersion: str = Form("2010-04-01"),
        CallStatus: str = Form(""),
        Direction: str = Form(""),
        From: str = Form(""),
        Caller: str = Form(""),
        CallerCity: Optional[str] = Form(None),
        CallerState: Optional[str] = Form(None),
        CallerZip: Optional[str] = Form(None),
        CallerCountry: Optional[str] = Form(None),
        FromCity: Optional[str] = Form(None),
        FromState: Optional[str] = Form(None),
        FromZip: Optional[str] = Form(None),
        FromCountry: Optional[str] = Form(None),
        To: str = Form(""),
        Called: str = Form(""),
        CalledCity: Optional[str] = Form(None),
        CalledState: Optional[str] = Form(None),
        CalledZip: Optional[str] = Form(None),
        CalledCountry: Optional[str] = Form(None),
        ToCity: Optional[str] = Form(None),
        ToState: Optional[str] = Form(None),
        ToZip: Optional[str] = Form(None),
        ToCountry: Optional[str] = Form(None),
        StirVerstat: Optional[str] = Form(None),
        CallToken: Optional[str] = Form(None),
    ) -> "CallWebhookInput":
        """
        Create CallWebhookInput from FastAPI Form fields.

        Usage:
            @app.post("/twilio/voice")
            async def webhook(data: CallWebhookInput = Depends(CallWebhookInput.as_form)):
                ...
        """
        return cls(
            CallSid=CallSid,
            AccountSid=AccountSid,
            ApiVersion=ApiVersion,
            CallStatus=CallStatus,
            Direction=Direction,
            From=From,
            Caller=Caller,
            CallerCity=CallerCity,
            CallerState=CallerState,
            CallerZip=CallerZip,
            CallerCountry=CallerCountry,
            FromCity=FromCity,
            FromState=FromState,
            FromZip=FromZip,
            FromCountry=FromCountry,
            To=To,
            Called=Called,
            CalledCity=CalledCity,
            CalledState=CalledState,
            CalledZip=CalledZip,
            CalledCountry=CalledCountry,
            ToCity=ToCity,
            ToState=ToState,
            ToZip=ToZip,
            ToCountry=ToCountry,
            StirVerstat=StirVerstat,
            CallToken=CallToken,
        )
