import asyncio
import os
from typing import Optional, Union, Iterator, AsyncIterator, List, Any

import boto3

from vision_agents.core.tts.tts import TTS as BaseTTS
from getstream.video.rtc.track_util import PcmData, AudioFormat


class TTS(BaseTTS):
    """AWS Polly Text-to-Speech implementation.

    Follows AWS Polly SynthesizeSpeech API:
    - OutputFormat is set to 'pcm' (signed 16-bit little-endian, mono)
    - SampleRate must be one of {'8000','16000'} for PCM
    - TextType can be 'text' or 'ssml' (auto-detected unless overridden)
    - Optional Engine ('standard' or 'neural'), LanguageCode, LexiconNames

    Credentials are resolved via standard AWS SDK chain (env vars, profiles, roles).
    """

    def __init__(
        self,
        *,
        region_name: Optional[str] = None,
        voice_id: str = "Joanna",
        text_type: Optional[str] = "text",  # 'text' | 'ssml'
        engine: Optional[str] = None,  # 'standard' | 'neural'
        language_code: Optional[str] = None,
        lexicon_names: Optional[List[str]] = None,
        client: Optional[Any] = None,
    ) -> None:
        super().__init__(provider_name="aws_polly")
        self.region_name = (
            region_name
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "us-east-1"
        )
        self.voice_id = voice_id

        if engine is not None and engine not in ("standard", "neural"):
            raise ValueError("engine must be 'standard' or 'neural' if provided")
        if text_type is not None and text_type not in ("text", "ssml"):
            raise ValueError("text_type must be 'text' or 'ssml' if provided")

        self.text_type = text_type
        self.engine = engine
        self.language_code = language_code
        self.lexicon_names = lexicon_names
        self._client = client

    @property
    async def client(self):
        if self._client is None:
            self._client = await asyncio.to_thread(
                lambda: boto3.client("polly", region_name=self.region_name)
            )
        return self._client

    async def stream_audio(
        self, text: str, *_, **__
    ) -> Union[PcmData, Iterator[PcmData], AsyncIterator[PcmData]]:
        """Synthesize the entire speech to a single PCM buffer.

        Returns PcmData with s16 format and the configured sample rate.
        """

        client = await self.client

        params = {
            "Text": text,
            "OutputFormat": "pcm",
            "VoiceId": self.voice_id,
            "SampleRate": "16000",
            "TextType": self.text_type,
        }

        if self.engine is not None:
            params["Engine"] = self.engine
        if self.language_code is not None:
            params["LanguageCode"] = self.language_code
        if self.lexicon_names:
            params["LexiconNames"] = self.lexicon_names  # type: ignore[assignment]

        # Wrap both the API call and stream read in a thread to avoid blocking
        def _synthesize_and_read():
            resp = client.synthesize_speech(**params)
            return resp["AudioStream"].read()

        audio_bytes = await asyncio.to_thread(_synthesize_and_read)

        return PcmData.from_bytes(
            audio_bytes, sample_rate=16000, channels=1, format=AudioFormat.S16
        )

    async def stop_audio(self) -> None:
        return None
