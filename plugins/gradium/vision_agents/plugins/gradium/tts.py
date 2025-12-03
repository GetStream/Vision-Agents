"""Gradium Text-to-Speech implementation using the official Gradium SDK."""

import logging
import os
from typing import Optional

import gradium
from getstream.video.rtc.track_util import PcmData, AudioFormat

from vision_agents.core import tts

logger = logging.getLogger(__name__)

# Default voice ID from Gradium docs
DEFAULT_VOICE_ID = "YTpq7expH9539ERJ"

# Gradium TTS PCM output specs
GRADIUM_TTS_SAMPLE_RATE = 48000
GRADIUM_TTS_CHANNELS = 1


class _TTSAsyncIterator:
    """Async iterator wrapper for Gradium TTS stream."""

    def __init__(self, client, setup, text):
        self._client = client
        self._setup = setup
        self._text = text
        self._stream = None

    def __aiter__(self):
        return self

    async def __anext__(self) -> PcmData:
        # Initialize stream on first iteration
        if self._stream is None:
            tts_stream = await gradium.speech.tts_stream(
                self._client,
                setup=self._setup,
                text=self._text,
            )
            self._stream = tts_stream.iter_bytes()

        try:
            audio_chunk = await self._stream.__anext__()
            return PcmData.from_bytes(
                audio_chunk,
                sample_rate=GRADIUM_TTS_SAMPLE_RATE,
                channels=GRADIUM_TTS_CHANNELS,
                format=AudioFormat.S16,
            )
        except StopAsyncIteration:
            raise


class TTS(tts.TTS):
    """
    Gradium Text-to-Speech implementation using the official Gradium SDK.

    Gradium provides low-latency, high-quality text-to-speech with support for
    multiple languages (English, French, German, Spanish, Portuguese),
    custom voice cloning, and speed control.

    Reference: https://gradium.ai/api_docs.html#tag/Documentation/Text-to-Speech-(TTS)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: str = DEFAULT_VOICE_ID,
        model_name: str = "default",
        speed: float = 0.0,
        client: Optional[gradium.client.GradiumClient] = None,
    ):
        """
        Initialize Gradium TTS.

        Args:
            api_key: Gradium API key. If not provided, uses GRADIUM_API_KEY env var.
            voice_id: Voice ID from Gradium voice library or custom voice.
                Defaults to Gradium's default voice.
            model_name: TTS model to use. Defaults to "default".
            speed: Speed control via padding_bonus. Range: -4.0 (faster) to 4.0 (slower).
                Defaults to 0.0 (normal speed).
            client: Optional pre-configured GradiumClient instance.
        """
        super().__init__(provider_name="gradium")

        self.voice_id = voice_id
        self.model_name = model_name
        self.speed = speed

        if client is not None:
            self.client = client
        else:
            api_key = api_key or os.environ.get("GRADIUM_API_KEY")
            if api_key:
                self.client = gradium.client.GradiumClient(api_key=api_key, base_url = "https://us.api.gradium.ai/api/")
            else:
                # Will use GRADIUM_API_KEY env var automatically
                self.client = gradium.client.GradiumClient()

    async def stream_audio(self, text: str, *_, **kwargs) -> _TTSAsyncIterator:
        """
        Convert text to speech using Gradium SDK streaming.

        Audio is streamed as PCM at 48kHz, 16-bit signed integer, mono.

        Args:
            text: The text to convert to speech.
            **kwargs: Additional arguments (voice_id, speed can be overridden).

        Returns:
            Async iterator yielding PcmData chunks.
        """
        # Allow overriding voice and speed per call
        voice_id = kwargs.get("voice_id", self.voice_id)
        speed = kwargs.get("speed", self.speed)

        # Build setup parameters
        setup = gradium.speech.TTSSetup(
            model_name=self.model_name,
            voice_id=voice_id,
            output_format="pcm",
        )

        # Add speed control if non-zero
        if speed != 0.0:
            setup["json_config"] = {"padding_bonus": speed}

        # Return async iterator that will stream from Gradium
        return _TTSAsyncIterator(self.client, setup, text)

    async def stop_audio(self) -> None:
        """
        Stop audio playback.

        This is a no-op for Gradium TTS as the agent manages playback.
        """
        logger.debug("Gradium TTS stop requested (no-op)")
