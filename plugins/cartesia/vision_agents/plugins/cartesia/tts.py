from __future__ import annotations

import logging
import os
from typing import Optional, cast, AsyncIterator, Iterator

from cartesia import AsyncCartesia
from cartesia.tts import (
    OutputFormat_RawParams,
    TtsRequestIdSpecifierParams,
    TtsRequestEmbeddingSpecifierParams,
)

from vision_agents.core import tts
from getstream.video.rtc.track_util import PcmData, AudioFormat

logger = logging.getLogger(__name__)


class TTS(tts.TTS):
    """Text-to-Speech plugin backed by the Cartesia Sonic model."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = "sonic-3",
        voice_id: str | None = "6ccbfb76-1fc6-48f7-b71d-91ac6298247b",
        sample_rate: int = 16000,
        client: Optional[AsyncCartesia] = None,
    ) -> None:
        """Create a new Cartesia TTS instance.

        Args:
            api_key: Cartesia API key – falls back to ``CARTESIA_API_KEY`` env var.
            model_id: Which model to use (default ``sonic-3``).
            voice_id: Cartesia voice ID. When ``None`` the model default is used.
            sample_rate: PCM sample-rate you want back (must match output track).
        """

        super().__init__()

        self.api_key = api_key or os.getenv("CARTESIA_API_KEY")
        if not self.api_key:
            raise ValueError("CARTESIA_API_KEY env var or api_key parameter required")

        self.client = (
            client if client is not None else AsyncCartesia(api_key=self.api_key)
        )
        self.model_id = model_id
        # Ensure voice_id is always provided for API typing and calls
        self.voice_id: str = (
            voice_id if voice_id is not None else "6ccbfb76-1fc6-48f7-b71d-91ac6298247b"
        )
        self.sample_rate = sample_rate

    async def stream_audio(
        self, text: str, *_, **__
    ) -> PcmData | Iterator[PcmData] | AsyncIterator[PcmData]:  # noqa: D401
        """Generate speech and return a stream of PcmData."""

        output_format: OutputFormat_RawParams = {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": self.sample_rate,
        }

        # ``/tts/bytes`` is the lowest-latency endpoint and returns an *async*
        # iterator when used with ``AsyncCartesia``. Each item yielded is
        # a raw ``bytes`` / ``bytearray`` containing PCM samples (new Cartesia SDK)

        voice_param: (
            TtsRequestIdSpecifierParams | TtsRequestEmbeddingSpecifierParams
        ) = cast(TtsRequestIdSpecifierParams, {"id": self.voice_id})

        response = self.client.tts.bytes(
            model_id=self.model_id,
            transcript=text,
            output_format=output_format,
            voice=voice_param,
        )

        return PcmData.from_response(
            response, sample_rate=self.sample_rate, channels=1, format=AudioFormat.S16
        )

    async def stop_audio(self) -> None:
        """
        Clears the queue and stops playing audio.
        This method can be used manually or under the hood in response to turn events.

        Returns:
            None
        """
        logger.info("🎤 Cartesia TTS stop requested (no-op)")
