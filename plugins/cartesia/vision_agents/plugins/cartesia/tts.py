from __future__ import annotations

import logging
import os
from typing import Optional, cast

from cartesia import AsyncCartesia
from cartesia.tts import (
    OutputFormat_RawParams,
    TtsRequestIdSpecifierParams,
    TtsRequestEmbeddingSpecifierParams,
)

from vision_agents.core import tts
from getstream.video.rtc.audio_track import AudioStreamTrack


class TTS(tts.TTS):
    """Text-to-Speech plugin backed by the Cartesia Sonic model."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = "sonic-2",
        voice_id: str | None = "f9836c6e-a0bd-460e-9d3c-f7299fa60f94",
        sample_rate: int = 16000,
        client: Optional[AsyncCartesia] = None,
    ) -> None:
        """Create a new Cartesia TTS instance.

        Args:
            api_key: Cartesia API key – falls back to ``CARTESIA_API_KEY`` env var.
            model_id: Which model to use (default ``sonic-2``).
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
            voice_id if voice_id is not None else "f9836c6e-a0bd-460e-9d3c-f7299fa60f94"
        )
        self.sample_rate = sample_rate

    def get_required_framerate(self) -> int:
        """Get the required framerate for Cartesia TTS."""
        return self.sample_rate

    def get_required_stereo(self) -> bool:
        """Get whether Cartesia TTS requires stereo audio."""
        return False  # Cartesia returns mono audio

    def set_output_track(self, track: AudioStreamTrack) -> None:  # noqa: D401
        if track.framerate != self.sample_rate:
            raise TypeError(
                f"Track framerate {track.framerate} ≠ expected {self.sample_rate}"
            )
        super().set_output_track(track)

    async def stream_audio(self, text: str, *_, **__) -> bytes:  # noqa: D401
        """Generate speech and yield raw PCM chunks."""

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

        async def _audio_chunk_stream():  # noqa: D401
            async for chunk in response:
                yield bytes(chunk)

        return _audio_chunk_stream()

    async def stop_audio(self) -> None:
        """
        Clears the queue and stops playing audio.
        This method can be used manually or under the hood in response to turn events.

        Returns:
            None
        """
        try:
            (await self.track.flush(),)
            logging.info("🎤 Stopping audio track for TTS")
            return
        except Exception as e:
            logging.error(f"Error flushing audio track: {e}")
