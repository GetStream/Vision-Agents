import os
from typing import Optional

from openai import AsyncOpenAI

from vision_agents.core.tts.tts import TTS as BaseTTS
from getstream.video.rtc.track_util import PcmData, AudioFormat

# Gandr returns pcm as headerless 16-bit signed little-endian mono at 24 kHz,
# so the bytes can be wrapped directly with no decode step.
SAMPLE_RATE = 24_000

# The Gandr API accepts at most 2000 characters per request.
MAX_INPUT_CHARS = 2000

GANDR_BASE_URL = "https://tts.gandr.ai/v1"


class TTS(BaseTTS):
    """Gandr Text-to-Speech implementation.

    Gandr exposes an OpenAI compatible speech endpoint, so this plugin drives
    it with the ``openai`` client pointed at ``https://tts.gandr.ai/v1``.
    Keys are available at https://gandr.ai.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "tts-1",
        voice: str = "gandr-mia",
        base_url: str = GANDR_BASE_URL,
        client: Optional[AsyncOpenAI] = None,
    ) -> None:
        """Initialize the Gandr TTS service.

        Args:
            api_key: Gandr API key. Falls back to the ``GANDR_API_KEY`` env var.
            model: Model name (default ``tts-1``).
            voice: Gandr voice, one of ``gandr-mia``, ``gandr-ava``,
                ``gandr-jenny``, ``gandr-dane``, ``gandr-leo``, ``gandr-lewis``.
            base_url: Gandr API base URL. Must use HTTPS.
            client: Optionally pass in your own ``AsyncOpenAI`` client. When
                set, ``api_key`` and ``base_url`` are ignored.
        """
        super().__init__(provider_name="gandr")
        self._owns_client = client is None
        if client is None:
            if not base_url.startswith("https://"):
                raise ValueError("base_url must use HTTPS")
            api_key = api_key or os.environ.get("GANDR_API_KEY")
            if not api_key:
                raise ValueError("GANDR_API_KEY env var or api_key parameter required")
            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.client = client
        self.model = model
        self.voice = voice

    async def stream_audio(self, text: str, *_, **__) -> PcmData:
        """Synthesize the entire speech to a single PCM buffer.

        Base TTS handles resampling and event emission.
        """
        if len(text) > MAX_INPUT_CHARS:
            raise ValueError(
                f"Gandr accepts at most {MAX_INPUT_CHARS} characters per "
                f"request, got {len(text)}"
            )

        resp = await self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format="pcm",
        )

        return PcmData.from_bytes(
            resp.content, sample_rate=SAMPLE_RATE, channels=1, format=AudioFormat.S16
        )

    async def stop_audio(self) -> None:
        # No internal playback queue; agent manages output track
        return None

    async def close(self) -> None:
        """Close the underlying HTTP client and release resources."""
        try:
            if self._owns_client:
                await self.client.close()
        finally:
            await super().close()
