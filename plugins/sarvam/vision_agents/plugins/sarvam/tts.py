"""Sarvam AI Text-to-Speech via WebSocket streaming.

Docs: https://docs.sarvam.ai/api-reference-docs/api-guides-tutorials/text-to-speech/streaming-api

The WebSocket stays open across ``stream_audio`` calls to avoid per-call
connection overhead. Text is sent as a JSON message; audio chunks arrive as
base64-encoded PCM which we decode into ``PcmData``.
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

WS_BASE_URL = "wss://api.sarvam.ai/text-to-speech/ws"

SUPPORTED_MODELS = {"bulbul:v2", "bulbul:v3"}


class TTS(tts.TTS):
    """Sarvam AI streaming Text-to-Speech.

    Keeps a persistent WebSocket open across synthesis calls. Sends a config
    message on first connect, then text + flush per ``stream_audio`` call.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "bulbul:v3",
        language: str = "hi-IN",
        speaker: str = "shubh",
        sample_rate: int = 24000,
        pace: Optional[float] = None,
        pitch: Optional[float] = None,
        loudness: Optional[float] = None,
        temperature: Optional[float] = None,
        enable_preprocessing: bool = True,
        idle_timeout: float = 1.5,
    ) -> None:
        """Initialize Sarvam TTS.

        Args:
            api_key: Sarvam API key. Falls back to ``SARVAM_API_KEY`` env var.
            model: TTS model. Defaults to ``bulbul:v3``.
            language: Target language code (e.g. ``hi-IN``, ``en-IN``).
            speaker: Speaker voice id (e.g. ``shubh``, ``anushka``).
            sample_rate: Output sample rate in Hz. Defaults to 24000.
            pace: Speech pace. Range depends on model
                (bulbul:v3 supports 0.5–2.0).
            pitch: Speech pitch. Only supported on bulbul:v2.
            loudness: Speech loudness. Only supported on bulbul:v2.
            temperature: Sampling temperature. Only supported on bulbul:v3.
            enable_preprocessing: Normalize mixed-language / numeric text.
            idle_timeout: Seconds of silence from the server before we treat
                a synthesis as complete. Sarvam does not always emit an
                explicit completion event, so this bounds ``stream_audio``.
        """
        super().__init__(provider_name="sarvam")

        if model not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported Sarvam TTS model '{model}'. "
                f"Expected one of: {sorted(SUPPORTED_MODELS)}"
            )

        self._api_key = api_key or os.environ.get("SARVAM_API_KEY")
        if not self._api_key:
            raise ValueError(
                "SARVAM_API_KEY env var or api_key parameter required for Sarvam TTS"
            )

        self.model = model
        self.language = language
        self.speaker = speaker
        self.sample_rate = sample_rate
        self.pace = pace
        self.pitch = pitch
        self.loudness = loudness
        self.temperature = temperature
        self.enable_preprocessing = enable_preprocessing
        self._idle_timeout = idle_timeout

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._lock = asyncio.Lock()
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        """Open the persistent WebSocket connection."""
        await self._ensure_connection()

    async def close(self) -> None:
        """Close the WebSocket and release the aiohttp session."""
        await self._reset_connection()
        await super().close()

    async def stream_audio(
        self, text: str, *_: Any, **__: Any
    ) -> AsyncIterator[PcmData]:
        """Stream TTS audio chunks for ``text`` over the persistent WebSocket.

        Returns:
            Async iterator yielding ``PcmData`` chunks.
        """
        self._stop_event.clear()
        async with self._lock:
            ws = await self._ensure_connection()
            await ws.send_str(json.dumps({"type": "text", "data": {"text": text}}))
            await ws.send_str(json.dumps({"type": "flush"}))
        return self._receive_audio(ws)

    async def stop_audio(self) -> None:
        """Cancel any in-flight synthesis and tear down the connection."""
        self._stop_event.set()
        if self._ws is not None and not self._ws.closed:
            try:
                await self._ws.send_str(json.dumps({"type": "cancel"}))
            except (aiohttp.ClientError, ConnectionError):
                pass
        # Easiest way to flush the server-side queue is a reconnect.
        await self._reset_connection()

    async def _ensure_connection(self) -> aiohttp.ClientWebSocketResponse:
        if self._ws is not None and not self._ws.closed:
            return self._ws

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

        url = f"{WS_BASE_URL}?model={self.model}"
        headers = {"api-subscription-key": self._api_key or ""}
        ws = await self._session.ws_connect(url, headers=headers)

        config: dict[str, Any] = {
            "target_language_code": self.language,
            "speaker": self.speaker,
            "speech_sample_rate": self.sample_rate,
            "enable_preprocessing": self.enable_preprocessing,
        }
        if self.pace is not None:
            config["pace"] = self.pace
        if self.pitch is not None:
            config["pitch"] = self.pitch
        if self.loudness is not None:
            config["loudness"] = self.loudness
        if self.temperature is not None:
            config["temperature"] = self.temperature

        await ws.send_str(json.dumps({"type": "config", "data": config}))
        self._ws = ws
        logger.debug("Sarvam TTS websocket connected at %dHz", self.sample_rate)
        return ws

    async def _reset_connection(self) -> None:
        if self._ws is not None and not self._ws.closed:
            try:
                await self._ws.close()
            except (aiohttp.ClientError, ConnectionError):
                logger.debug("Error closing Sarvam TTS websocket")
        self._ws = None

        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _receive_audio(
        self, ws: aiohttp.ClientWebSocketResponse
    ) -> AsyncIterator[PcmData]:
        """Yield PcmData chunks until flushed, cancelled, idle, or disconnected.

        Sarvam does not always send an explicit completion event after the
        final audio chunk, so we also treat a short idle gap (no message
        within ``self._idle_timeout`` seconds) as end-of-stream.
        """
        while True:
            if self._stop_event.is_set():
                break
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=self._idle_timeout)
            except asyncio.TimeoutError:
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
                logger.warning("Sarvam TTS sent non-JSON text: %s", msg.data)
                continue

            msg_type = data.get("type", "")
            if msg_type in ("audio", "audio_chunk"):
                payload = data.get("data") or {}
                b64_audio = payload.get("audio") or data.get("audio")
                if not b64_audio:
                    continue
                audio_bytes = base64.b64decode(b64_audio)
                yield PcmData.from_bytes(
                    audio_bytes,
                    sample_rate=self.sample_rate,
                    channels=1,
                    format=AudioFormat.S16,
                )
            elif msg_type in ("flushed", "complete", "done"):
                break
            elif msg_type == "error":
                logger.error("Sarvam TTS error: %s", data)
                break
