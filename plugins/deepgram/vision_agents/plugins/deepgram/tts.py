import logging
import os
from typing import AsyncIterator, Optional

from deepgram import AsyncDeepgramClient
from getstream.video.rtc.track_util import PcmData, AudioFormat

from vision_agents.core import tts

logger = logging.getLogger(__name__)


class TTS(tts.TTS):
    """
    Deepgram Text-to-Speech implementation using Aura model.

    Uses the Deepgram Speak API with streaming response.

    References:
    - https://developers.deepgram.com/docs/text-to-speech
    - https://developers.deepgram.com/docs/tts-models
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "aura-2-thalia-en",
        sample_rate: int = 16000,
        client: Optional[AsyncDeepgramClient] = None,
    ):
        """
        Initialize Deepgram TTS.

        Args:
            api_key: Deepgram API key. If not provided, will use DEEPGRAM_API_KEY env var.
            model: Voice model to use. Defaults to "aura-2-thalia-en".
                   See https://developers.deepgram.com/docs/tts-models for available voices.
            sample_rate: Audio sample rate in Hz. Defaults to 16000.
            client: Optional pre-configured AsyncDeepgramClient instance.
        """
        super().__init__(provider_name="deepgram")

        if not api_key:
            api_key = os.environ.get("DEEPGRAM_API_KEY")

        if client is not None:
            self.client = client
        else:
            if api_key:
                self.client = AsyncDeepgramClient(api_key=api_key)
            else:
                self.client = AsyncDeepgramClient()

        self.model = model
        self.sample_rate = sample_rate

    async def stream_audio(self, text: str, *_, **__) -> AsyncIterator[PcmData]:
        """
        Convert text to speech using Deepgram's Speak API.

        Args:
            text: The text to convert to speech.

        Returns:
            An async iterator of PcmData audio chunks.
        """
        # Use the Deepgram speak API with streaming response
        response = self.client.speak.v1.audio.generate(
            text=text,
            model=self.model,
            encoding="linear16",
            sample_rate=self.sample_rate,
            container="none",  # Raw PCM, no container
        )

        return PcmData.from_response(
            response,
            sample_rate=self.sample_rate,
            channels=1,
            format=AudioFormat.S16,
        )

    async def stop_audio(self) -> None:
        """
        Stop audio playback.

        This is a no-op for Deepgram TTS as each stream_audio call
        creates its own request.
        """
        pass
