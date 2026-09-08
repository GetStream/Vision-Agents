"""Smallest AI Text-to-Speech via WebSocket streaming (Lightning).

Docs:
    - Streaming: https://docs.smallest.ai/waves/documentation/text-to-speech-lightning/http-vs-streaming-vs-web-sockets
    - Voices: https://docs.smallest.ai/waves/documentation/text-to-speech-lightning/overview

Smallest AI's streaming API is one-request-per-connection: a fresh
WebSocket is opened for each ``stream_audio`` call and closed after the
``complete`` frame, rather than kept open across calls.
"""

import asyncio
import base64
import json
import logging
import os
from typing import Any, AsyncIterator, Optional

import aiohttp
from getstream.video.rtc.track_util import AudioFormat, PcmData
from vision_agents.core import tts

logger = logging.getLogger(__name__)

WS_URL = "wss://api.smallest.ai/waves/v1/tts/live"

SUPPORTED_MODELS = {"lightning_v3.1", "lightning_v3.1_pro"}


class SmallestTTSError(Exception):
    """Raised when Smallest AI TTS returns an error frame over WebSocket."""


class TTS(tts.TTS):
    """Smallest AI Lightning streaming Text-to-Speech."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: str = "magnus",
        model: str = "lightning_v3.1",
        sample_rate: int = 24000,
        speed: Optional[float] = None,
    ) -> None:
        """Initialize Smallest AI TTS.

        Args:
            api_key: Smallest AI API key. Falls back to ``SMALLEST_API_KEY`` env var.
            voice_id: Catalog or cloned voice id (e.g. ``magnus``, ``voice_*``).
            model: TTS model. ``lightning_v3.1`` (default, supports voice cloning)
                or ``lightning_v3.1_pro`` (premium pool, English/Hindi only).
            sample_rate: Output sample rate in Hz. Defaults to 24000.
            speed: Speech rate multiplier, 0.5-2.0.
        """
        super().__init__(provider_name="smallest")

        if model not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported Smallest AI TTS model '{model}'. "
                f"Expected one of: {sorted(SUPPORTED_MODELS)}"
            )

        self._api_key = api_key or os.environ.get("SMALLEST_API_KEY")
        if not self._api_key:
            raise ValueError(
                "SMALLEST_API_KEY env var or api_key parameter required for Smallest AI TTS"
            )

        self.voice_id = voice_id
        self.model = model
        self.sample_rate = sample_rate
        self.speed = speed

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._stop_event = asyncio.Event()

    async def stream_audio(
        self, text: str, *_: Any, **__: Any
    ) -> AsyncIterator[PcmData]:
        """Stream TTS audio chunks for ``text`` over a fresh WebSocket connection.

        Returns:
            Async iterator yielding ``PcmData`` chunks.
        """

        async def _stream() -> AsyncIterator[PcmData]:
            self._stop_event.clear()
            ws = await self._connect()
            self._ws = ws
            request: dict[str, Any] = {
                "text": text,
                "voice_id": self.voice_id,
                "model": self.model,
                "sample_rate": self.sample_rate,
            }
            if self.speed is not None:
                request["speed"] = self.speed
            await ws.send_str(json.dumps(request))
            async for chunk in self._receive_audio(ws):
                yield chunk

        return _stream()

    async def stop_audio(self) -> None:
        """Close the in-flight WebSocket connection to cancel synthesis."""
        self._stop_event.set()
        if self._ws is not None and not self._ws.closed:
            await self._ws.close()

    async def close(self) -> None:
        """Stop any in-flight synthesis and release the aiohttp session."""
        await super().close()
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _connect(self) -> aiohttp.ClientWebSocketResponse:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "X-Source": "getstream",
        }
        ws = await self._session.ws_connect(WS_URL, headers=headers)
        self._on_connected()
        logger.debug("Smallest AI TTS websocket connected at %dHz", self.sample_rate)
        return ws

    async def _receive_audio(
        self, ws: aiohttp.ClientWebSocketResponse
    ) -> AsyncIterator[PcmData]:
        """Yield PcmData chunks until the ``complete`` frame, stop, or disconnect."""
        try:
            async for msg in ws:
                if self._stop_event.is_set():
                    break
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                    aiohttp.WSMsgType.ERROR,
                ):
                    break
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    logger.warning("Smallest AI TTS sent non-JSON text: %s", msg.data)
                    continue

                status = data.get("status", "")
                if status == "chunk":
                    b64_audio = (data.get("data") or {}).get("audio")
                    if not b64_audio:
                        continue
                    audio_bytes = base64.b64decode(b64_audio)
                    yield PcmData.from_bytes(
                        audio_bytes,
                        sample_rate=self.sample_rate,
                        channels=1,
                        format=AudioFormat.S16,
                    )
                elif status == "complete":
                    break
                elif status == "error":
                    error_msg = data.get("message") or "Smallest AI TTS error"
                    raise SmallestTTSError(str(error_msg))
        finally:
            if not ws.closed:
                await ws.close()
            if self._ws is ws:
                self._ws = None
            self._on_disconnected()
