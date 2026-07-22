import logging
import os
from typing import AsyncIterator, Iterator, Optional

from getstream.video.rtc.track_util import AudioFormat, PcmData
from vision_agents.core import tts

from speechify import AsyncSpeechify

logger = logging.getLogger(__name__)

# Speechify streams raw PCM as 16-bit signed little-endian mono at 24 kHz
# (audio/L16; rate=24000; channels=1) when the ``audio/pcm`` Accept type is used.
SAMPLE_RATE = 24000


class TTS(tts.TTS):
    """Text-to-Speech plugin backed by the Speechify streaming API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: str = "geffen_32",
        model: str = "simba-3.2",
        language: Optional[str] = None,
        client: Optional[AsyncSpeechify] = None,
    ) -> None:
        """Create a new Speechify TTS instance.

        Args:
            api_key: Speechify API key – falls back to ``SPEECHIFY_API_KEY`` env var.
            voice_id: Speechify voice ID (default ``geffen_32``).
            model: Which model to use (default ``simba-3.2``).
            language: Optional ISO 639-1 + ISO 3166-1 language tag (e.g. ``en-US``).
        """

        super().__init__(provider_name="speechify")

        self.api_key = api_key or os.getenv("SPEECHIFY_API_KEY")
        if not self.api_key:
            raise ValueError("SPEECHIFY_API_KEY env var or api_key parameter required")

        self.client = (
            client if client is not None else AsyncSpeechify(token=self.api_key)
        )
        self.voice_id = voice_id
        self.model = model
        self.language = language

    async def stream_audio(
        self, text: str, *_, **__
    ) -> PcmData | Iterator[PcmData] | AsyncIterator[PcmData]:
        """Generate speech and return a stream of PcmData."""

        audio_stream = self.client.audio.stream(
            accept="audio/pcm",
            input=text,
            voice_id=self.voice_id,
            model=self.model,
            language=self.language,
        )

        return PcmData.from_response(
            audio_stream,
            sample_rate=SAMPLE_RATE,
            channels=1,
            format=AudioFormat.S16,
        )

    async def stop_audio(self) -> None:
        """Clears the queue and stops playing audio.

        Speechify uses a one-shot streaming HTTP request, so there is nothing to
        cancel server-side. This method can be used manually or under the hood in
        response to turn events.
        """
        logger.debug("🎤 Speechify TTS stop requested (no-op)")

    async def close(self) -> None:
        """Close the underlying HTTP client and release resources."""
        try:
            await self.client._client_wrapper.httpx_client.httpx_client.aclose()
        finally:
            await super().close()
