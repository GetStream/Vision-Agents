import asyncio
import logging
import os
from typing import AsyncContextManager, AsyncGenerator, AsyncIterator, Optional

import numpy as np
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

# Silence padding before/after audio keeps the audio track active
# across idle↔active transitions, preventing codec pops.
_SILENCE_PAD_MS = 150

# Pause between consecutive utterances to mimic natural sentence spacing.
_INTER_UTTERANCE_PAUSE_MS = 200

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
        api_key: Optional[str] = None,
        model: str = "aura-2-thalia-en",
        sample_rate: int = 16000,
        client: Optional[AsyncDeepgramClient] = None,
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
        elif api_key:
            self.client = AsyncDeepgramClient(api_key=api_key)
        else:
            self.client = AsyncDeepgramClient()

        self.model = model
        self.sample_rate = sample_rate

        self._socket: Optional[AsyncV1SocketClient] = None
        self._connection_ctx: Optional[AsyncContextManager[AsyncV1SocketClient]] = None
        self._receiver: Optional[AsyncGenerator[PcmData, None]] = None
        self._needs_drain = False
        self._stop_event = asyncio.Event()
        self._effective_rate = sample_rate

    def _pick_sample_rate(self) -> int:
        """Choose the best sample rate to request from Deepgram.

        Matches the downstream output rate when possible to avoid
        resampling entirely. Falls back to the configured rate.
        """
        if self._desired_sample_rate in _SUPPORTED_RATES:
            return self._desired_sample_rate
        return self.sample_rate

    async def _ensure_connection(self) -> AsyncV1SocketClient:
        """Open the websocket if not already connected."""
        if self._socket is not None:
            return self._socket

        self._effective_rate = self._pick_sample_rate()
        ctx = self.client.speak.v1.connect(
            model=self.model,
            encoding="linear16",
            sample_rate=str(self._effective_rate),
        )
        self._connection_ctx = ctx
        self._socket = await ctx.__aenter__()
        logger.debug("Deepgram TTS websocket connected at %dHz", self._effective_rate)
        return self._socket

    async def _reset_connection(self) -> None:
        """Tear down the current connection so the next call reopens it."""
        self._receiver = None
        self._needs_drain = False
        if self._connection_ctx is not None:
            try:
                await self._connection_ctx.__aexit__(None, None, None)
            except (websockets.exceptions.WebSocketException, OSError):
                pass
        self._socket = None
        self._connection_ctx = None

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

        if self._receiver is not None:
            try:
                await self._receiver.aclose()
            except RuntimeError:
                pass
            self._receiver = None

        if self._needs_drain:
            await self._drain(socket)
            self._needs_drain = False

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

        gen = self._receive_audio(socket)
        self._receiver = gen
        return gen

    def _silence(self, ms: int) -> PcmData:
        """Create a silent PcmData chunk of the given duration."""
        rate = self._effective_rate
        return PcmData(
            sample_rate=rate,
            format=AudioFormat.S16,
            samples=np.zeros(int(rate * ms / 1000), dtype=np.int16),
            channels=1,
        )

    async def _receive_audio(
        self, socket: AsyncV1SocketClient
    ) -> AsyncGenerator[PcmData, None]:
        """Yield PcmData for each websocket message until flushed.

        Silence padding before the first and after the last audio chunk
        keeps the audio track active across transitions, preventing
        codec startup/shutdown pops.
        """
        rate = self._effective_rate
        is_first = True
        try:
            async for message in socket:
                if self._stop_event.is_set():
                    break
                if isinstance(message, bytes):
                    if is_first:
                        yield self._silence(_SILENCE_PAD_MS)
                        is_first = False
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
            # Silence tail + inter-utterance pause
            yield self._silence(_SILENCE_PAD_MS + _INTER_UTTERANCE_PAUSE_MS)
        finally:
            self._receiver = None

    async def stop_audio(self) -> None:
        """Send Clear to cancel in-flight synthesis on the server."""
        self._stop_event.set()
        if self._socket is not None:
            try:
                await self._socket.send_clear()
                self._needs_drain = True
            except (websockets.exceptions.ConnectionClosed, ConnectionError):
                await self._reset_connection()

    async def close(self) -> None:
        """Close the persistent websocket connection."""
        if self._receiver is not None:
            await self._receiver.aclose()
            self._receiver = None
        if self._socket is not None and self._connection_ctx is not None:
            try:
                await self._socket.send_close()
            except (websockets.exceptions.WebSocketException, OSError) as exc:
                logger.warning("Error sending close to Deepgram TTS: %s", exc)
            try:
                await self._connection_ctx.__aexit__(None, None, None)
            except (websockets.exceptions.WebSocketException, OSError) as exc:
                logger.warning("Error closing Deepgram TTS connection: %s", exc)
            finally:
                self._socket = None
                self._connection_ctx = None
        await super().close()
