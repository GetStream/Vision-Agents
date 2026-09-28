import asyncio
import logging
import os
from contextlib import AsyncExitStack
from typing import AsyncIterator

import websockets.exceptions
from deepgram import AsyncDeepgramClient
from deepgram.speak.v2.socket_client import AsyncV2SocketClient
from deepgram.speak.v2.types import (
    SpeakV2Speak,
    SpeakV2SpeechInterrupted,
    SpeakV2SpeechMetadata,
    SpeakV2Warning,
)
from getstream.video.rtc.track_util import AudioFormat, PcmData
from vision_agents.core import tts

logger = logging.getLogger(__name__)


# Sample rates supported by Deepgram Flux TTS websocket API.
_SUPPORTED_RATES = {8000, 16000, 24000, 32000, 44100, 48000}


class TTS(tts.TTS):
    """Deepgram Text-to-Speech using Flux TTS (`/v2/speak`).

    Keeps a persistent websocket connection open across synthesis calls
    to avoid per-call connection overhead and audio discontinuities.

    References:
    - https://developers.deepgram.com/docs/flux-tts/overview
    - https://developers.deepgram.com/docs/flux-tts/voices
    """

    # This implementation accepts partial text detlas
    streaming = True

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "flux-haley-en",
        sample_rate: int = 16000,
        speed: float | None = None,
        client: AsyncDeepgramClient | None = None,
    ):
        """Initialize Deepgram TTS.

        Args:
            api_key: Deepgram API key. If not provided, will use DEEPGRAM_API_KEY env var.
            model: Flux voice model. Defaults to "flux-haley-en".
            sample_rate: Audio sample rate in Hz. Defaults to 16000.
            speed: Optional speech-rate multiplier (0.85–1.15 in 0.05 steps).
            client: Optional pre-configured AsyncDeepgramClient instance.
        """
        super().__init__(provider_name="deepgram")

        if not api_key:
            api_key = os.environ.get("DEEPGRAM_API_KEY")

        if sample_rate not in _SUPPORTED_RATES:
            raise ValueError(
                f"Deepgram TTS supports sample rates {sorted(_SUPPORTED_RATES)}; got {sample_rate}"
            )

        if model.startswith("aura"):
            raise ValueError(
                "Deepgram TTS uses Flux models (e.g. flux-haley-en, flux-kit-en). "
                "Aura model strings are not supported. See "
                "https://developers.deepgram.com/docs/flux-tts/voices"
            )

        if client is not None:
            self.client = client
        else:
            self.client = AsyncDeepgramClient(api_key=api_key)

        self.model = model
        self.sample_rate = sample_rate
        self.speed = speed

        self._socket: AsyncV2SocketClient | None = None
        self._exit_stack = AsyncExitStack()

        self._generation = 0
        self._stop_event = asyncio.Event()

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
        self._on_disconnected()
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
            await socket.send_speak(SpeakV2Speak(text=text))
            await socket.send_flush()
        except (websockets.exceptions.ConnectionClosed, ConnectionError):
            logger.warning("Deepgram TTS websocket dropped, reconnecting")
            await self._reset_connection()
            socket = await self._ensure_connection()
            await socket.send_speak(SpeakV2Speak(text=text))
            await socket.send_flush()

        self._generation += 1
        return self._receive_audio(socket, self._generation)

    async def stop_audio(self) -> None:
        """Send Interrupt to cancel in-flight synthesis on the server."""
        self._stop_event.set()
        if self._socket is not None:
            try:
                await self._socket.send_interrupt()
            except (websockets.exceptions.ConnectionClosed, ConnectionError):
                await self._reset_connection()

    async def _ensure_connection(self) -> AsyncV2SocketClient:
        """Open the websocket if not already connected."""
        if self._socket is not None:
            return self._socket

        socket = await self._exit_stack.enter_async_context(
            self.client.speak.v2.connect(
                model=self.model,
                encoding="linear16",
                sample_rate=str(self.sample_rate),
                speed=self.speed,
            )
        )
        self._socket = socket
        self._on_connected()
        logger.debug("Deepgram TTS websocket connected at %dHz", self.sample_rate)
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

    async def _drain(self, socket: AsyncV2SocketClient) -> None:
        """Consume any stale messages left on the websocket after interrupts.

        Uses a short timeout rather than waiting for a specific sentinel,
        because Deepgram may not send SpeechInterrupted if nothing was active.
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
        self, socket: AsyncV2SocketClient, generation: int
    ) -> AsyncIterator[PcmData]:
        """Yield PcmData for each websocket message until SpeechMetadata."""
        async for message in socket:
            if self._stop_event.is_set() or self._generation != generation:
                break
            if isinstance(message, bytes):
                yield PcmData.from_bytes(
                    message,
                    sample_rate=self.sample_rate,
                    channels=1,
                    format=AudioFormat.S16,
                )
            elif isinstance(message, (SpeakV2SpeechMetadata, SpeakV2SpeechInterrupted)):
                break
            elif isinstance(message, SpeakV2Warning):
                logger.warning("Deepgram TTS warning: %s", message)
