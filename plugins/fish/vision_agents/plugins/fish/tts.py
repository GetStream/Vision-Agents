import logging
import os
from typing import AsyncIterator, Iterator, Optional

from fish_audio_sdk import Session, TTSRequest
from vision_agents.core import tts
from getstream.video.rtc.track_util import PcmData, AudioFormat

logger = logging.getLogger(__name__)


class TTS(tts.TTS):
    """
    Fish Audio Text-to-Speech implementation.

    Fish Audio provides high-quality, multilingual text-to-speech synthesis with
    support for voice cloning via reference audio.


    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        reference_id: Optional[str] = "03397b4c4be74759b72533b663fbd001",
        base_url: Optional[str] = None,
        client: Optional[Session] = None,
    ):
        """
        Initialize the Fish Audio TTS service.

        Args:
            api_key: Fish Audio API key. If not provided, the FISH_AUDIO_API_KEY
                    environment variable will be used.
            reference_id: Optional reference voice ID to use for synthesis.
            base_url: Optional custom API endpoint.
            client: Optionally pass in your own instance of the Fish Audio Session.
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
            self.client = Session(api_key, base_url=base_url)
        else:
            self.client = Session(api_key)

        self.reference_id = reference_id

    async def stream_audio(
        self, text: str, *_, **kwargs
    ) -> PcmData | Iterator[PcmData] | AsyncIterator[PcmData]:
        """
        Convert text to speech using Fish Audio API.

        Args:
            text: The text to convert to speech.
            **kwargs: Additional arguments to pass to TTSRequest (e.g., references).

        Returns:
            An async iterator of audio chunks as bytes.
        """
        # Build the TTS request
        tts_request_kwargs = {"text": text}

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

        # Stream audio from Fish Audio; let PcmData normalize response types
        stream = self.client.tts.awaitable(tts_request)
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
