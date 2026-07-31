import logging
import os
from typing import Any, AsyncIterator, Iterator, Literal, Optional, Protocol, cast

from fish_audio_sdk import Session, TTSRequest
from getstream.video.rtc.track_util import AudioFormat, PcmData
from vision_agents.core import tts

logger = logging.getLogger(__name__)

FishTTSModel = Literal[
    "s2.1-pro",
    "s2.1-pro-free",
    "s2-pro",
    "s1",
    "s1-mini",
    "speech-1.5",
    "speech-1.6",
]


class _TTSAwaitable(Protocol):
    def __call__(
        self, request: TTSRequest, *, backend: FishTTSModel
    ) -> AsyncIterator[bytes]: ...


class TTS(tts.TTS):
    """
    Fish Audio Text-to-Speech implementation.

    Fish Audio provides high-quality, multilingual text-to-speech synthesis with
    support for voice cloning via reference audio and multiple backend models.

    Supported models:
        - s2.1-pro: Recommended production model with improved quality, latency,
                    and throughput (default)
        - s2.1-pro-free: Same model for testing and prototyping, without
                         production latency or availability guarantees
        - s2-pro: Previous-generation S2 model
        - s1: Previous-generation model
        - s1-mini: Lightweight S1 model
        - speech-1.5: Deprecated legacy model
        - speech-1.6: Deprecated legacy model
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        reference_id: Optional[str] = "9a9cf47702da476aa4629e2506d4a857",
        base_url: Optional[str] = None,
        client: Optional[Session] = None,
        model: FishTTSModel = "s2.1-pro",
    ):
        """
        Initialize the Fish Audio TTS service.

        Args:
            api_key: Fish Audio API key. If not provided, the FISH_AUDIO_API_KEY
                    environment variable will be used.
            reference_id: Optional reference voice ID to use for synthesis.
            base_url: Optional custom API endpoint.
            client: Optionally pass in your own instance of the Fish Audio Session.
            model: Backend model to use. Defaults to "s2.1-pro", Fish Audio's
                   recommended production model. Use "s2.1-pro-free" for
                   testing and prototyping without production guarantees.
        """
        super().__init__(provider_name="fish")

        if not api_key:
            # Support both env names for compatibility
            api_key = os.environ.get("FISH_API_KEY") or os.environ.get(
                "FISH_AUDIO_API_KEY"
            )

        if client is not None:
            self.client = client
        elif base_url:
            if not api_key:
                raise ValueError("api_key is required when base_url is provided")
            self.client = Session(api_key, base_url=base_url)
        else:
            if not api_key:
                raise ValueError("api_key is required")
            self.client = Session(api_key)

        self.reference_id = reference_id
        self.model: FishTTSModel = model

    async def stream_audio(
        self, text: str, *_, **kwargs: Any
    ) -> PcmData | Iterator[PcmData] | AsyncIterator[PcmData]:
        """
        Convert text to speech using Fish Audio API.

        Args:
            text: The text to convert to speech. When using an S2 model,
                  you can include inline control tags like [laugh], [whisper],
                  [super happy] for fine-grained prosody control.
            **kwargs: Additional arguments to pass to TTSRequest (e.g., references).

        Returns:
            An async iterator of audio chunks as bytes.
        """
        # Build the TTS request
        tts_request_kwargs: dict[str, Any] = {"text": text}

        # Add reference_id if configured
        if self.reference_id:
            tts_request_kwargs["reference_id"] = self.reference_id

        # Allow overriding via kwargs (e.g., for dynamic reference audio)
        tts_request_kwargs.update(kwargs)

        tts_request = TTSRequest(
            format="pcm",
            sample_rate=16000,
            normalize=True,
            **tts_request_kwargs,
        )

        # The SDK sends backend directly as the model header, but its model
        # literal currently lags the public API. Keep that compatibility
        # assertion at the third-party boundary instead of weakening our model type.
        tts_awaitable = cast(_TTSAwaitable, self.client.tts.awaitable)
        stream = tts_awaitable(tts_request, backend=self.model)
        return PcmData.from_response(
            stream, sample_rate=16000, channels=1, format=AudioFormat.S16
        )

    async def stop_audio(self) -> None:
        """
        Clears the queue and stops playing audio.

        This method can be used manually or under the hood in response to turn events.

        Returns:
            None
        """
        # No internal output track to flush; agent manages playback
        logger.info("🎤 Fish TTS stop requested (no-op)")
