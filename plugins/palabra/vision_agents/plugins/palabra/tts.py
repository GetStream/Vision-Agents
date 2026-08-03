"""Palabra AI Text-to-Speech via the realtime WebSocket API.

Docs: https://platform.palabra.ai/docs/text-to-speech/realtime-tts

A single WebSocket stays open across ``stream_audio`` calls: the session is
initialised once, then every utterance is a ``text`` message tagged with its own
``generation_id``. Audio chunks echo that id, so chunks left over from a
cancelled utterance are dropped without reconnecting.
"""

import asyncio
import base64
import json
import logging
import os
import uuid
from typing import AsyncIterator, Optional

import websockets
import websockets.exceptions
from getstream.video.rtc.track_util import AudioFormat, PcmData
from vision_agents.core import tts

logger = logging.getLogger(__name__)

WS_URL_EU = "wss://stream.palabra.ai/tts-api/v1/text-to-speech/stream"
WS_URL_US = "wss://stream.us.palabra.ai/tts-api/v1/text-to-speech/stream"

# Palabra rejects text messages above this length, so longer utterances are
# split across several messages.
MAX_TEXT_LENGTH = 1024

CANCEL_MESSAGE = json.dumps({"type": "cancel"})


class PalabraTTSError(Exception):
    """Raised when Palabra reports an error over the WebSocket."""

    def __init__(self, code: str, desc: str) -> None:
        super().__init__(f"{code}: {desc}")
        self.code = code
        self.desc = desc


def _split_text(text: str) -> list[str]:
    """Split ``text`` into messages of at most ``MAX_TEXT_LENGTH`` characters."""
    if len(text) <= MAX_TEXT_LENGTH:
        return [text]

    chunks: list[str] = []
    remaining = text
    while len(remaining) > MAX_TEXT_LENGTH:
        split_at = remaining.rfind(" ", 0, MAX_TEXT_LENGTH + 1)
        if split_at <= 0:
            split_at = MAX_TEXT_LENGTH
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


class TTS(tts.TTS):
    """Palabra AI streaming Text-to-Speech."""

    streaming = True

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: str = "default_low",
        language: str = "en",
        model: str = "auto",
        sample_rate: int = 24000,
        speed: Optional[float] = None,
        deaccent_strength: Optional[float] = None,
        ws_url: str = WS_URL_EU,
    ) -> None:
        """Initialize Palabra TTS.

        Args:
            api_key: Palabra API key. Falls back to ``PALABRA_API_KEY`` env var.
            voice_id: Voice to synthesize with – ``default_low``, ``default_high``
                or the id of a cloned voice.
            language: BCP-47 language code of the text (e.g. ``en``, ``de``, ``pt-eu``).
            model: TTS model id. ``auto`` lets Palabra pick.
            sample_rate: Output sample rate in Hz (8000–48000).
            speed: Speech speed multiplier (0.0–2.0). ``None`` uses the server default.
            deaccent_strength: Accent reduction for cloned voices (0.0–1.0).
                ``None`` uses the server default.
            ws_url: Palabra TTS endpoint. Defaults to the EU region; pass
                ``WS_URL_US`` for the US region.
        """
        super().__init__(provider_name="palabra")

        api_key = api_key or os.getenv("PALABRA_API_KEY")
        if not api_key:
            raise ValueError(
                "PALABRA_API_KEY env var or api_key parameter required for Palabra TTS"
            )
        if not 8000 <= sample_rate <= 48000:
            raise ValueError(
                f"Palabra TTS sample_rate must be between 8000 and 48000 Hz; got {sample_rate}"
            )
        if speed is not None and not 0.0 <= speed <= 2.0:
            raise ValueError(
                f"Palabra TTS speed must be between 0.0 and 2.0; got {speed}"
            )

        self.voice_id = voice_id
        self.language = language
        self.model = model
        self.sample_rate = sample_rate

        self._api_key = api_key
        self._ws_url = ws_url

        voice_options: dict[str, object] = {"voice_id": voice_id}
        if speed is not None:
            voice_options["speed"] = speed
        if deaccent_strength is not None:
            voice_options["deaccent_strength"] = deaccent_strength
        # Settings apply to the whole session and cannot be changed once sent
        # (the server answers CONFLICT), so the payload is built once here.
        self._init_message = json.dumps(
            {
                "type": "init",
                "language": language,
                "model": model,
                "voice_options": voice_options,
                "output": {"format": "pcm", "sample_rate": sample_rate},
            }
        )

        self._websocket: websockets.ClientConnection | None = None
        self._pending_recv: asyncio.Future[bytes] | None = None
        self._generation = 0

    async def start(self) -> None:
        """Open the WebSocket up front so the first utterance skips the handshake."""
        await self._ensure_connection()

    async def close(self) -> None:
        """Close the WebSocket and release resources."""
        await self._reset_connection()
        self._on_disconnected()
        await super().close()

    async def stream_audio(self, text: str, *_, **__) -> AsyncIterator[PcmData]:
        """Synthesize ``text`` over the persistent WebSocket.

        Args:
            text: The text to convert to speech.

        Returns:
            Async iterator yielding ``PcmData`` chunks.
        """
        generation_id = uuid.uuid4().hex
        self._generation += 1
        generation = self._generation

        try:
            websocket = await self._send_text(text, generation_id)
        except (websockets.exceptions.WebSocketException, OSError):
            logger.warning("Palabra TTS websocket dropped; reconnecting")
            await self._reset_connection()
            websocket = await self._send_text(text, generation_id)

        return self._receive_audio(websocket, generation_id, generation)

    async def stop_audio(self) -> None:
        """Cancel in-flight synthesis. The session stays open for the next utterance."""
        self._generation += 1

        # Palabra does not confirm the cancel, so a reader parked in recv() would
        # sit there until the *next* utterance produced a frame. Cancelling the
        # pending read releases it now; websockets guarantees the connection
        # stays usable and no message is lost. Detaching it first is how the
        # reader tells our cancel apart from its own task being cancelled.
        pending_recv = self._pending_recv
        if pending_recv is not None:
            self._pending_recv = None
            pending_recv.cancel()

        websocket = self._websocket
        if websocket is None:
            return
        try:
            await websocket.send(CANCEL_MESSAGE)
        except (websockets.exceptions.WebSocketException, OSError):
            await self._reset_connection()

    async def _send_text(
        self, text: str, generation_id: str
    ) -> websockets.ClientConnection:
        websocket = await self._ensure_connection()
        chunks = _split_text(text)
        last = len(chunks) - 1
        for index, chunk in enumerate(chunks):
            await websocket.send(
                json.dumps(
                    {
                        "type": "text",
                        "text": chunk,
                        "generation_id": generation_id,
                        "is_eos": index == last,
                    }
                )
            )
        return websocket

    async def _receive_audio(
        self,
        websocket: websockets.ClientConnection,
        generation_id: str,
        generation: int,
    ) -> AsyncIterator[PcmData]:
        while self._generation == generation:
            # decode=False keeps the frame as bytes, which json parses directly.
            recv = asyncio.ensure_future(websocket.recv(decode=False))
            self._pending_recv = recv
            try:
                message = await recv
            except asyncio.CancelledError:
                # stop_audio() detaches the read before cancelling it; anything
                # still attached means our own task is being cancelled, which
                # must propagate.
                if self._pending_recv is not recv:
                    return
                raise
            except (websockets.exceptions.ConnectionClosed, OSError):
                await self._reset_connection()
                raise
            finally:
                if self._pending_recv is recv:
                    self._pending_recv = None

            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                logger.warning("Skipping non-JSON Palabra TTS websocket message")
                continue

            message_type = data.get("message_type")
            payload = data.get("data") or {}

            if message_type == "error":
                raise PalabraTTSError(
                    payload.get("code", "UNKNOWN_ERROR"), payload.get("desc", "")
                )
            if message_type != "audio_chunk":
                continue
            # Chunks of a cancelled or abandoned utterance the server is still
            # draining onto the shared socket.
            if payload.get("generation_id") != generation_id:
                continue

            audio = payload.get("audio")
            if audio:
                yield PcmData.from_bytes(
                    base64.b64decode(audio),
                    sample_rate=self.sample_rate,
                    channels=1,
                    format=AudioFormat.S16,
                )
            if payload.get("last_chunk"):
                return

    async def _ensure_connection(self) -> websockets.ClientConnection:
        websocket = self._websocket
        if websocket is not None and websocket.state is websockets.State.OPEN:
            return websocket

        await self._reset_connection()
        websocket = await websockets.connect(
            self._ws_url,
            additional_headers={"Authorization": f"Bearer {self._api_key}"},
            # Base64-encoded PCM barely compresses; deflating every audio frame
            # only adds CPU work and latency on the playback path.
            compression=None,
        )
        await websocket.send(self._init_message)
        self._websocket = websocket
        self._on_connected()
        logger.debug(
            "Palabra TTS websocket connected at %dHz with voice %s",
            self.sample_rate,
            self.voice_id,
        )
        return websocket

    async def _reset_connection(self) -> None:
        websocket = self._websocket
        self._websocket = None
        if websocket is None:
            return
        try:
            await websocket.close()
        except (websockets.exceptions.WebSocketException, OSError):
            logger.debug("Error closing Palabra TTS websocket")
