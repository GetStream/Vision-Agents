import asyncio
import logging
import os
from contextlib import AsyncExitStack
from typing import AsyncIterator

import websockets.exceptions
from deepgram import AsyncDeepgramClient
from deepgram.speak.v1.socket_client import AsyncV1SocketClient
from deepgram.speak.v1.types import (
    SpeakV1Cleared,
    SpeakV1Flushed,
    SpeakV1Text,
    SpeakV1Warning,
)
from getstream.video.rtc.track_util import AudioFormat, PcmData
from vision_agents.core import tts

logger = logging.getLogger(__name__)


# Sample rates supported by Deepgram TTS websocket API.
_SUPPORTED_RATES = {8000, 16000, 24000, 48000}


class TTS(tts.TTS):
    """Deepgram Text-to-Speech using the WebSocket streaming API.

    Keeps a persistent websocket connection open across synthesis calls
    to avoid per-call connection overhead and audio discontinuities.

    References:
    - https://developers.deepgram.com/docs/text-to-speech
    - https://developers.deepgram.com/docs/tts-models
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "aura-2-thalia-en",
        sample_rate: int = 16000,
        client: AsyncDeepgramClient | None = None,
    ):
        """Initialize Deepgram TTS.

        Args:
            api_key: Deepgram API key. If not provided, will use DEEPGRAM_API_KEY env var.
            model: Voice model to use. Defaults to "aura-2-thalia-en".
            sample_rate: Audio sample rate in Hz. Defaults to 16000.
            client: Optional pre-configured AsyncDeepgramClient instance.
        """
        super().__init__(provider_name="deepgram")

        if not api_key:
            api_key = os.environ.get("DEEPGRAM_API_KEY")

        if client is not None:
            self.client = client
        else:
            self.client = AsyncDeepgramClient(api_key=api_key)

        self.model = model
        self.sample_rate = sample_rate

        self._socket: AsyncV1SocketClient | None = None
        self._exit_stack = AsyncExitStack()

        self._generation = 0
        self._stop_event = asyncio.Event()
        self._effective_rate = sample_rate

    async def start(self) -> None:
        await self._ensure_connection()

    async def close(self) -> None:
        """Close the persistent websocket connection."""
        if self._socket is not None:
            try:
                await self._socket.send_close()
            except (websockets.exceptions.WebSocketException, OSError) as exc:
                logger.warning("Error sending close to Deepgram TTS: %s", exc)
        await self._reset_connection()
        await super().close()

    async def stream_audio(self, text: str, *_, **__) -> AsyncIterator[PcmData]:
        """Stream TTS audio chunks over a persistent websocket.

        Args:
            text: The text to convert to speech.

        Returns:
            An async iterator of PcmData audio chunks.
        """
        try:
            socket = await self._ensure_connection()
        except (websockets.exceptions.ConnectionClosed, ConnectionError):
            await self._reset_connection()
            socket = await self._ensure_connection()

        if self._stop_event.is_set():
            await self._drain(socket)
        self._stop_event.clear()

        try:
            await socket.send_text(SpeakV1Text(text=text))
            await socket.send_flush()
        except (websockets.exceptions.ConnectionClosed, ConnectionError):
            logger.warning("Deepgram TTS websocket dropped, reconnecting")
            await self._reset_connection()
            socket = await self._ensure_connection()
            await socket.send_text(SpeakV1Text(text=text))
            await socket.send_flush()

        self._generation += 1
        return self._receive_audio(socket, self._generation)

    async def stop_audio(self) -> None:
        """Send Clear to cancel in-flight synthesis on the server."""
        self._stop_event.set()
        if self._socket is not None:
            try:
                await self._socket.send_clear()
            except (websockets.exceptions.ConnectionClosed, ConnectionError):
                await self._reset_connection()

    async def _ensure_connection(self) -> AsyncV1SocketClient:
        """Open the websocket if not already connected."""
        if self._socket is not None:
            return self._socket

        if self._desired_sample_rate in _SUPPORTED_RATES:
            self._effective_rate = self._desired_sample_rate
        else:
            self._effective_rate = self.sample_rate

        socket = await self._exit_stack.enter_async_context(
            self.client.speak.v1.connect(
                model=self.model,
                encoding="linear16",
                sample_rate=str(self._effective_rate),
            )
        )
        self._socket = socket
        logger.debug("Deepgram TTS websocket connected at %dHz", self._effective_rate)
        return socket

    async def _reset_connection(self) -> None:
        """Tear down the current connection so the next call reopens it."""
        self._generation += 1
        try:
            await self._exit_stack.aclose()
        finally:
            self._stop_event.clear()
            self._exit_stack = AsyncExitStack()
            self._socket = None

    async def _drain(self, socket: AsyncV1SocketClient) -> None:
        """Consume any stale messages left on the websocket after interrupts.

        Uses a short timeout rather than waiting for a specific sentinel,
        because Deepgram may not send Cleared if nothing was active.
        """
        while True:
            try:
                await asyncio.wait_for(socket.recv(), timeout=0.05)
            except TimeoutError:
                break
            except websockets.exceptions.ConnectionClosed:
                await self._reset_connection()
                break

    async def _receive_audio(
        self, socket: AsyncV1SocketClient, generation: int
    ) -> AsyncIterator[PcmData]:
        """
        Yield PcmData for each websocket message until flushed.
        """
        rate = self._effective_rate
        async for message in socket:
            if self._stop_event.is_set() or self._generation != generation:
                break
            if isinstance(message, bytes):
                yield PcmData.from_bytes(
                    message,
                    sample_rate=rate,
                    channels=1,
                    format=AudioFormat.S16,
                )
            elif isinstance(message, SpeakV1Flushed):
                break
            elif isinstance(message, SpeakV1Cleared):
                continue
            elif isinstance(message, SpeakV1Warning):
                logger.warning("Deepgram TTS warning: %s", message)
