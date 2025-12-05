import asyncio
import logging
import os
from typing import AsyncIterator, Optional

from deepgram import AsyncDeepgramClient
from getstream.video.rtc.track_util import PcmData, AudioFormat

from vision_agents.core import tts

logger = logging.getLogger(__name__)


class TTS(tts.TTS):
    """
    Deepgram Text-to-Speech implementation using Aura model.

    Uses WebSocket streaming for low-latency audio generation.

    References:
    - https://developers.deepgram.com/docs/tts-websocket
    - https://developers.deepgram.com/docs/streaming-text-to-speech
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "aura-2-thalia-en",
        sample_rate: int = 16000,
        client: Optional[AsyncDeepgramClient] = None,
    ):
        """
        Initialize Deepgram TTS.

        Args:
            api_key: Deepgram API key. If not provided, will use DEEPGRAM_API_KEY env var.
            model: Voice model to use. Defaults to "aura-2-thalia-en".
                   See https://developers.deepgram.com/docs/tts-models for available voices.
            sample_rate: Audio sample rate in Hz. Defaults to 16000.
            client: Optional pre-configured AsyncDeepgramClient instance.
        """
        super().__init__(provider_name="deepgram")

        if not api_key:
            api_key = os.environ.get("DEEPGRAM_API_KEY")

        if client is not None:
            self.client = client
        else:
            if api_key:
                self.client = AsyncDeepgramClient(api_key=api_key)
            else:
                self.client = AsyncDeepgramClient()

        self.model = model
        self.sample_rate = sample_rate

    async def stream_audio(self, text: str, *_, **__) -> AsyncIterator[PcmData]:
        """
        Convert text to speech using Deepgram's WebSocket API.

        Args:
            text: The text to convert to speech.

        Returns:
            An async iterator of PcmData audio chunks.
        """
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        error_holder: list[Exception] = []

        def on_binary_data(_self, data: bytes, **kwargs):
            """Handle incoming audio data from Deepgram."""
            audio_queue.put_nowait(data)

        def on_error(_self, error, **kwargs):
            """Handle errors from Deepgram."""
            logger.error("Deepgram TTS WebSocket error: %s", error)
            error_holder.append(Exception(f"Deepgram TTS error: {error}"))
            audio_queue.put_nowait(None)

        def on_close(_self, **kwargs):
            """Handle connection close."""
            audio_queue.put_nowait(None)

        # Connect to Deepgram TTS WebSocket
        options = {
            "model": self.model,
            "encoding": "linear16",
            "sample_rate": self.sample_rate,
        }

        connection = self.client.speak.asyncwebsocket.v("1")
        await connection.start(options)

        # Register event handlers
        connection.on("AudioData", on_binary_data)
        connection.on("Error", on_error)
        connection.on("Close", on_close)

        async def generate_audio() -> AsyncIterator[PcmData]:
            try:
                # Send text to be synthesized
                await connection.send_text(text)

                # Signal that we're done sending text
                await connection.flush()

                # Collect audio chunks until connection closes
                while True:
                    chunk = await audio_queue.get()
                    if chunk is None:
                        break

                    if error_holder:
                        raise error_holder[0]

                    # Convert raw bytes to PcmData
                    pcm = PcmData.from_bytes(
                        chunk,
                        sample_rate=self.sample_rate,
                        channels=1,
                        format=AudioFormat.S16,
                    )
                    yield pcm

            finally:
                await connection.finish()

        return generate_audio()

    async def stop_audio(self) -> None:
        """
        Stop audio playback.

        This is a no-op for Deepgram TTS as each stream_audio call
        creates its own connection.
        """
        pass
