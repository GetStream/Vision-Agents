import base64
import io
import json
import logging
import os
from typing import AsyncIterator, Literal, Optional

import av
import httpx
from getstream.video.rtc.track_util import PcmData

from vision_agents.core import tts

logger = logging.getLogger(__name__)

INWORLD_API_BASE = "https://api.inworld.ai"


class TTS(tts.TTS):
    """
    Inworld AI Text-to-Speech implementation.
    Inworld AI provides high-quality text-to-speech synthesis with streaming support.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: str = "Dennis",
        model_id: Literal["inworld-tts-1", "inworld-tts-1-max"] = "inworld-tts-1",
        temperature: float = 1.1,
    ):
        """
        Initialize the Inworld AI TTS service.
        Args:
            api_key: Inworld AI API key. If not provided, the INWORLD_API_KEY
                    environment variable will be used.
            voice_id: The voice ID to use for synthesis (default: "Dennis").
            model_id: The model ID to use for synthesis. Options: "inworld-tts-1",
                     "inworld-tts-1-max" (default: "inworld-tts-1").
            temperature: Determines the degree of randomness when sampling audio tokens.
                        Accepts values between 0 and 2. Default: 1.1.
        """
        super().__init__(provider_name="inworld")

        api_key = api_key or os.getenv("INWORLD_API_KEY")
        if not api_key:
            raise ValueError(
                "INWORLD_API_KEY environment variable must be set or api_key must be provided"
            )

        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.temperature = temperature
        self.base_url = INWORLD_API_BASE
        self.client = httpx.AsyncClient(timeout=60.0)

    async def stream_audio(self, text: str, *_, **__) -> AsyncIterator[PcmData]:
        """
        Convert text to speech using Inworld AI API.
        Args:
            text: The text to convert to speech (max 2,000 characters).
        Returns:
            An async iterator of audio chunks as PcmData objects.
        """
        url = f"{self.base_url}/tts/v1/voice:stream"

        credentials = f"Basic {self.api_key}"
        headers = {
            "Authorization": credentials,
            "Content-Type": "application/json",
        }

        payload = {
            "text": text,
            "voiceId": self.voice_id,
            "modelId": self.model_id,
            "audioConfig": {
                "temperature": self.temperature,
            },
        }

        async def _stream_audio() -> AsyncIterator[PcmData]:
            try:
                async with self.client.stream(
                    "POST", url, headers=headers, json=payload
                ) as response:
                    async for pcm in self._process_response(response):
                        yield pcm
            except httpx.HTTPStatusError as e:
                logger.error(
                    "Inworld AI API HTTP error: %s - %s",
                    e.response.status_code,
                    e.response.text,
                )
                raise
            except Exception as e:
                logger.error("Error streaming audio from Inworld AI: %s", e)
                raise

        # Return the async generator
        return _stream_audio()

    async def _process_response(
        self, response: httpx.Response
    ) -> AsyncIterator[PcmData]:
        # Check status before processing streaming response
        if response.status_code >= 400:
            error_text = await response.aread()
            error_msg = error_text.decode() if error_text else "Unknown error"
            logger.error(
                "Inworld AI API HTTP error: %s - %s",
                response.status_code,
                error_msg,
            )
            raise httpx.HTTPStatusError(
                f"HTTP {response.status_code}: {error_msg}",
                request=response.request,
                response=response,
            )

        async for line in response.aiter_lines():
            if not line.strip():
                continue

            try:
                data = json.loads(line)
                if "error" in data:
                    error_msg = data["error"].get("message", "Unknown error")
                    logger.error("Inworld AI API error: %s", error_msg)
                    continue

                if "result" in data and "audioContent" in data["result"]:
                    wav_bytes = base64.b64decode(data["result"]["audioContent"])

                    container = av.open(io.BytesIO(wav_bytes))
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

                        if pcm:
                            pcm = pcm.resample(
                                target_sample_rate=pcm.sample_rate,
                                target_channels=1,
                            ).to_int16()
                            yield pcm
            except json.JSONDecodeError as e:
                logger.warning("Failed to parse JSON line: %s", e)
                continue
            except Exception as e:
                logger.warning("Error processing audio chunk: %s", e)
                continue

    async def stop_audio(self) -> None:
        """
        Clears the queue and stops playing audio.
        This method can be used manually or under the hood in response to turn events.
        Returns:
            None
        """
        logger.info("🎤 Inworld AI TTS stop requested (no-op)")

    async def close(self) -> None:
        if self.client:
            await self.client.aclose()
