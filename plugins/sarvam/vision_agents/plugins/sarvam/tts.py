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

SUPPORTED_MODELS = {"bulbul:v2", "bulbul:v3-beta", "bulbul:v3"}

KEEPALIVE_INTERVAL_S = 20

MODEL_SPEAKER_COMPATIBILITY: dict[str, set[str]] = {
    "bulbul:v2": {
        "anushka",
        "manisha",
        "vidya",
        "arya",
        "abhilash",
        "karun",
        "hitesh",
    },
    "bulbul:v3-beta": {
        "shubh",
        "ritu",
        "rahul",
        "pooja",
        "simran",
        "kavya",
        "amit",
        "ratan",
        "rohan",
        "dev",
        "ishita",
        "shreya",
        "manan",
        "sumit",
        "priya",
        "aditya",
        "kabir",
        "neha",
        "varun",
        "roopa",
        "aayan",
        "ashutosh",
        "advait",
        "amelia",
        "sophia",
    },
    "bulbul:v3": {
        "shubh",
        "ritu",
        "rahul",
        "pooja",
        "simran",
        "kavya",
        "amit",
        "ratan",
        "rohan",
        "dev",
        "ishita",
        "shreya",
        "manan",
        "sumit",
        "priya",
        "aditya",
        "kabir",
        "neha",
        "varun",
        "roopa",
        "aayan",
        "ashutosh",
        "advait",
        "amelia",
        "sophia",
    },
}

MODELS_SUPPORTING_PITCH = {"bulbul:v2"}
MODELS_SUPPORTING_LOUDNESS = {"bulbul:v2"}
MODELS_SUPPORTING_TEMPERATURE = {"bulbul:v3-beta", "bulbul:v3"}


class SarvamTTSError(Exception):
    """Raised when Sarvam TTS returns an error message over WebSocket."""


class TTS(tts.TTS):
    """Sarvam AI streaming Text-to-Speech.

    Keeps a persistent WebSocket open across synthesis calls. Sends a config
    message on first connect, then text + flush.
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
        idle_timeout: float = 5.0,
    ) -> None:
        """Initialize Sarvam TTS.

        Args:
            api_key: Sarvam API key. Falls back to ``SARVAM_API_KEY`` env var.
            model: TTS model. Defaults to ``bulbul:v3``.
            language: Target language code (e.g. ``hi-IN``, ``en-IN``).
            speaker: Speaker voice id (e.g. ``shubh``, ``anushka``).
            sample_rate: Output sample rate in Hz. Defaults to 24000.
            pace: Speech pace. Range depends on model
                (bulbul:v3 supports 0.5-2.0).
            pitch: Speech pitch. Only supported on bulbul:v2.
            loudness: Speech loudness. Only supported on bulbul:v2.
            temperature: Sampling temperature. Only supported on
                bulbul:v3 / bulbul:v3-beta.
            enable_preprocessing: Normalize mixed-language / numeric text.
            idle_timeout: Fallback seconds of server silence before treating
                synthesis as complete. Normally the server sends an explicit
                completion event; this is a safety net.
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

        compatible = MODEL_SPEAKER_COMPATIBILITY.get(model)
        if compatible is not None and speaker not in compatible:
            raise ValueError(
                f"Speaker '{speaker}' is not compatible with model '{model}'. "
                f"Compatible speakers: {sorted(compatible)}"
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
        self._keepalive_task: Optional[asyncio.Task[None]] = None

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

        async def _stream() -> AsyncIterator[PcmData]:
            self._stop_event.clear()
            async with self._lock:
                ws = await self._ensure_connection()
                await ws.send_str(json.dumps({"type": "text", "data": {"text": text}}))
                await ws.send_str(json.dumps({"type": "flush"}))
                async for chunk in self._receive_audio(ws):
                    yield chunk

        return _stream()

    async def stop_audio(self) -> None:
        """Cancel any in-flight synthesis and tear down the connection."""
        self._stop_event.set()
        if self._ws is not None and not self._ws.closed:
            try:
                await self._ws.send_str(json.dumps({"type": "cancel"}))
            except (aiohttp.ClientError, ConnectionError):
                pass
        await self._reset_connection()

    async def _ensure_connection(self) -> aiohttp.ClientWebSocketResponse:
        if self._ws is not None and not self._ws.closed:
            return self._ws

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

        url = f"{WS_BASE_URL}?model={self.model}&send_completion_event=true"
        headers = {"api-subscription-key": self._api_key or ""}
        ws = await self._session.ws_connect(url, headers=headers)

        config: dict[str, Any] = {
            "model": self.model,
            "target_language_code": self.language,
            "speaker": self.speaker,
            "speech_sample_rate": self.sample_rate,
            "enable_preprocessing": self.enable_preprocessing,
            "output_audio_codec": "linear16",
        }
        if self.pace is not None:
            config["pace"] = self.pace
        if self.pitch is not None and self.model in MODELS_SUPPORTING_PITCH:
            config["pitch"] = self.pitch
        if self.loudness is not None and self.model in MODELS_SUPPORTING_LOUDNESS:
            config["loudness"] = self.loudness
        if self.temperature is not None and self.model in MODELS_SUPPORTING_TEMPERATURE:
            config["temperature"] = self.temperature

        await ws.send_str(json.dumps({"type": "config", "data": config}))
        self._ws = ws
        self._start_keepalive()
        logger.debug("Sarvam TTS websocket connected at %dHz", self.sample_rate)
        return ws

    def _start_keepalive(self) -> None:
        self._stop_keepalive()
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    def _stop_keepalive(self) -> None:
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            self._keepalive_task = None

    async def _keepalive_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(KEEPALIVE_INTERVAL_S)
                if self._ws is not None and not self._ws.closed:
                    await self._ws.send_str(json.dumps({"type": "ping"}))
        except asyncio.CancelledError:
            pass
        except (aiohttp.ClientError, ConnectionError):
            logger.debug("Sarvam TTS keepalive send failed")

    async def _reset_connection(self) -> None:
        self._stop_keepalive()

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
        """Yield PcmData chunks until completion event, cancel, idle, or disconnect."""
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
            elif msg_type == "event":
                event_data = data.get("data") or {}
                if event_data.get("event_type") == "final":
                    break
            elif msg_type in ("flushed", "complete", "done"):
                break
            elif msg_type == "error":
                error_data = data.get("data") or {}
                error_msg = (
                    error_data.get("message") or data.get("error") or "Sarvam TTS error"
                )
                raise SarvamTTSError(str(error_msg))
