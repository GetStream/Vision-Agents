import base64
import io
import json
import logging
import os
from typing import AsyncIterator, Iterator, Optional

import av
import httpx
from vision_agents.core import tts
from getstream.video.rtc.track_util import PcmData, AudioFormat

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
        model_id: str = "inworld-tts-1",
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

        if not api_key:
            api_key = os.environ.get("INWORLD_API_KEY")
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

    async def stream_audio(
        self, text: str, *_, **__
    ) -> PcmData | Iterator[PcmData] | AsyncIterator[PcmData]:
        """
        Convert text to speech using Inworld AI API.

        Args:
            text: The text to convert to speech (max 2,000 characters).

        Returns:
            An async iterator of audio chunks as PcmData.
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
                async with self.client.stream("POST", url, headers=headers, json=payload) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue

                        try:
                            data = json.loads(line)
                            if "error" in data:
                                error_msg = data["error"].get("message", "Unknown error")
                                logger.error(f"Inworld AI API error: {error_msg}")
                                continue

                            if "result" in data and "audioContent" in data["result"]:
                                wav_bytes = base64.b64decode(data["result"]["audioContent"])
                                
                                # Decode WAV to PCM using PyAV
                                with av.open(io.BytesIO(wav_bytes)) as container:
                                    audio_stream = container.streams.audio[0]
                                    pcm = None
                                    
                                    for frame in container.decode(audio_stream):
                                        frame_pcm = PcmData.from_av_frame(frame)
                                        if pcm is None:
                                            pcm = frame_pcm
                                        else:
                                            pcm.append(frame_pcm)
                                    
                                    if pcm:
                                        pcm = pcm.resample(
                                            target_sample_rate=pcm.sample_rate,
                                            target_channels=1
                                        ).to_int16()
                                        yield pcm
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse JSON line: {e}")
                            continue
                        except Exception as e:
                            logger.warning(f"Error processing audio chunk: {e}")
                            continue

            except httpx.HTTPStatusError as e:
                logger.error(f"Inworld AI API HTTP error: {e.response.status_code} - {e.response.text}")
                raise
            except Exception as e:
                logger.error(f"Error streaming audio from Inworld AI: {e}")
                raise

        return _stream_audio()

    async def stop_audio(self) -> None:
        """
        Clears the queue and stops playing audio.

        This method can be used manually or under the hood in response to turn events.

        Returns:
            None
        """
        logger.info("🎤 Inworld AI TTS stop requested (no-op)")


    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - close HTTP client if we created it."""
        if self.client:
            await self.client.aclose()

