import os
from typing import Optional

from openai import AsyncOpenAI

from vision_agents.core.tts.tts import TTS as BaseTTS
from getstream.video.rtc.track_util import PcmData, AudioFormat


class TTS(BaseTTS):
    """OpenAI Text-to-Speech implementation.

    Uses OpenAI's TTS models to synthesize speech.
    Docs: https://platform.openai.com/docs/guides/text-to-speech
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini-tts",
        voice: str = "alloy",
        client: Optional[AsyncOpenAI] = None,
    ) -> None:
        super().__init__(provider_name="openai_tts")
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = client or AsyncOpenAI(api_key=api_key)
        self.model = model
        self.voice = voice

    async def stream_audio(self, text: str, *_, **__) -> PcmData:
        """Synthesize the entire speech to a single PCM buffer.

        Base TTS handles resampling and event emission.
        """
        resp = await self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format="pcm",
        )

        return PcmData.from_bytes(
            resp.content, sample_rate=24_000, channels=1, format=AudioFormat.S16
        )

    async def stop_audio(self) -> None:
        # No internal playback queue; agent manages output track
        return None
