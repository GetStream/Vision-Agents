from __future__ import annotations

import io
import logging
import os
from typing import AsyncIterator, Iterator, Optional

import av
import httpx
from getstream.video.rtc.track_util import AudioFormat, PcmData
from vision_agents.core import tts

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://dev.voice.ai"
DEFAULT_PCM_SAMPLE_RATE = 32000
VALID_AUDIO_FORMATS = {"pcm", "wav", "mp3"}


class TTS(tts.TTS):
    """Voice.ai Text-to-Speech implementation (HTTP streaming)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: Optional[str] = "d1bf0f33-8e0e-4fbf-acf8-45c3c6262513",
        audio_format: str = "pcm",
        model: Optional[str] = "voiceai-tts-v1-latest",
        language: Optional[str] = "en",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = 60.0,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """
        Initialize Voice.ai TTS.

        Args:
            api_key: Voice.ai API key. Falls back to VOICE_AI_API_KEY env var.
            voice_id: Voice ID to use. Defaults to voice called "Ellie".
            audio_format: Output format: "pcm" (streamed), "wav", or "mp3".
            model: Optional model ID to use (Available options: voiceai-tts-v1-latest, voiceai-tts-v1-2025-12-19, voiceai-tts-multilingual-v1-latest, voiceai-tts-multilingual-v1-2025-01-14). Defaults to "voiceai-tts-v1-latest".
            language: Optional language code (Available options: en, ca, sv, es, fr, de, it, pt, pl, ru, nl). Defaults to "en".
            temperature: Optional sampling temperature (range from 0.0-2.0). Defaults to 1.
            top_p: Optional top-p nucleus sampling (range from 0.0-1.0). Defaults to 0.8.
            base_url: Voice.ai API base URL.
            timeout_s: HTTP timeout in seconds.
            client: Optional pre-configured httpx.AsyncClient.
        """
        super().__init__(provider_name="voice_ai")

        api_key = api_key or os.getenv("VOICE_AI_API_KEY")
        if not api_key:
            raise ValueError(
                "VOICE_AI_API_KEY env var or api_key parameter is required"
            )

        voice_id = voice_id or os.getenv("VOICE_AI_VOICE_ID")
        if not voice_id:
            raise ValueError(
                "VOICE_AI_VOICE_ID env var or voice_id parameter is required"
            )

        if audio_format not in VALID_AUDIO_FORMATS:
            raise ValueError(
                f"Unsupported audio_format '{audio_format}'. "
                f"Expected one of: {sorted(VALID_AUDIO_FORMATS)}"
            )

        self.api_key = api_key
        self.voice_id = voice_id
        self.audio_format = audio_format
        self.model = model
        self.language = language
        self.temperature = temperature
        self.top_p = top_p
        self.base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self.client = (
            client if client is not None else httpx.AsyncClient(timeout=timeout_s)
        )

    def _build_payload(self, text: str) -> dict:
        payload: dict = {
            "text": text,
            "voice_id": self.voice_id,
            "audio_format": self.audio_format,
        }
        if self.model is not None:
            payload["model"] = self.model
        if self.language is not None:
            payload["language"] = self.language
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        return payload

    async def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        error_bytes = await response.aread()
        error_text = error_bytes.decode(errors="ignore") if error_bytes else ""
        logger.error(
            "Voice.ai TTS HTTP error %s: %s",
            response.status_code,
            error_text,
        )
        raise httpx.HTTPStatusError(
            f"HTTP {response.status_code}: {error_text}",
            request=response.request,
            response=response,
        )

    def _decode_to_pcm(self, audio_bytes: bytes) -> PcmData:
        container = av.open(io.BytesIO(audio_bytes))
        assert isinstance(container, av.container.InputContainer)
        with container:
            audio_stream = container.streams.audio[0]
            pcm: Optional[PcmData] = None
            for frame in container.decode(audio_stream):
                frame_pcm = PcmData.from_av_frame(frame)
                if pcm is None:
                    pcm = frame_pcm
                else:
                    pcm.append(frame_pcm)

            if pcm is None:
                raise ValueError("No audio frames decoded from Voice.ai response")

            pcm = pcm.resample(
                target_sample_rate=pcm.sample_rate,
                target_channels=1,
            ).to_int16()
            return pcm

    async def stream_audio(
        self, text: str, *_, **__
    ) -> PcmData | Iterator[PcmData] | AsyncIterator[PcmData]:
        """Convert text to speech using Voice.ai's streaming endpoint."""
        url = f"{self.base_url}/api/v1/tts/speech/stream"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = self._build_payload(text)

        if self.audio_format == "pcm":

            async def _stream_bytes() -> AsyncIterator[bytes]:
                async with self.client.stream(
                    "POST", url, headers=headers, json=payload
                ) as response:
                    await self._raise_for_status(response)
                    async for chunk in response.aiter_bytes():
                        if chunk:
                            yield chunk

            return PcmData.from_response(
                _stream_bytes(),
                sample_rate=DEFAULT_PCM_SAMPLE_RATE,
                channels=1,
                format=AudioFormat.S16,
            )

        async with self.client.stream(
            "POST", url, headers=headers, json=payload
        ) as response:
            await self._raise_for_status(response)
            data = await response.aread()

        return self._decode_to_pcm(data)

    async def stop_audio(self) -> None:
        """Stop audio playback (no-op for Voice.ai)."""
        logger.info("Voice.ai TTS stop requested (no-op)")

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()
        await super().close()
