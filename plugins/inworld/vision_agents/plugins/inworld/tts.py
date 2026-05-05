import asyncio
import base64
import io
import json
import logging
import os
import uuid
from importlib.metadata import PackageNotFoundError, version
from typing import AsyncIterator, Iterator, Literal

import av
import websockets
import websockets.exceptions
from getstream.video.rtc.track_util import AudioFormat, PcmData
from vision_agents.core import tts

logger = logging.getLogger(__name__)

INWORLD_WS_URL = "wss://api.inworld.ai/tts/v1/voice:streamBidirectional"
KEEPALIVE_INTERVAL_SECONDS = 60

try:
    _pkg_version = version("vision-agents-plugins-inworld")
except PackageNotFoundError:
    _pkg_version = "unknown"

USER_AGENT = f"vision-agents-plugins-inworld/{_pkg_version}"


def _decode_audio(data: bytes, target_sample_rate: int) -> PcmData | None:
    """Decode an audio chunk (WAV or raw PCM) into a PcmData object."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        if len(data) < 2:
            return None
        return PcmData.from_bytes(
            data, sample_rate=target_sample_rate, channels=1, format=AudioFormat.S16
        )

    container = av.open(io.BytesIO(data), format="wav", mode="r")
    try:
        frames = list(container.decode(audio=0))
    finally:
        container.close()

    if not frames:
        return None

    result = PcmData.from_av_frame(frames[0])
    for frame in frames[1:]:
        result = result.append(PcmData.from_av_frame(frame))

    if result.sample_rate != target_sample_rate:
        result = result.resample(
            target_sample_rate=target_sample_rate, target_channels=1
        ).to_int16()

    return result


class TTS(tts.TTS):
    """Inworld AI Text-to-Speech plugin using bidirectional WebSocket streaming."""

    streaming = True

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str = "Sarah",
        model_id: Literal[
            "inworld-tts-1.5-max",
            "inworld-tts-1.5-mini",
            "inworld-tts-1",
            "inworld-tts-1-max",
            "inworld-tts-2",
        ] = "inworld-tts-2",
        sample_rate: int = 16000,
        temperature: float = 1.1,
        speaking_rate: float | None = None,
        auto_mode: bool = True,
        apply_text_normalization: Literal["ON", "OFF"] | None = None,
        ws_url: str = INWORLD_WS_URL,
    ):
        """Initialize the Inworld AI WebSocket TTS service.

        Args:
            api_key: Inworld AI API key – falls back to ``INWORLD_API_KEY`` env var.
            voice_id: The voice ID to use for synthesis.
            model_id: The model to use for synthesis.
            sample_rate: Desired PCM sample rate in Hz.
            temperature: Randomness when sampling audio tokens (0–2).
            speaking_rate: Speech speed multiplier (0.5–1.5). ``None`` uses the server default.
            auto_mode: Whether Inworld should decide optimal flush behavior.
            apply_text_normalization: Optional text normalization behavior.
            ws_url: Inworld bidirectional WebSocket endpoint.
        """
        super().__init__(provider_name="inworld")

        api_key = api_key or os.getenv("INWORLD_API_KEY")
        if not api_key:
            raise ValueError(
                "INWORLD_API_KEY environment variable must be set or api_key must be provided"
            )

        self._api_key = api_key
        self._voice_id = voice_id
        self._model_id = model_id
        self._sample_rate = sample_rate
        self._temperature = temperature
        self._speaking_rate = speaking_rate
        self._auto_mode = auto_mode
        self._apply_text_normalization = apply_text_normalization
        self._ws_url = ws_url

        self._websocket: websockets.ClientConnection | None = None
        self._generation = 0
        self._stop_event = asyncio.Event()
        self._active_context_id: str | None = None
        self._keepalive_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Open the persistent Inworld WebSocket connection."""
        await self._ensure_connection()

    async def stream_audio(
        self, text: str, *_, **__
    ) -> PcmData | Iterator[PcmData] | AsyncIterator[PcmData]:
        """Convert text to speech over Inworld bidirectional WebSocket."""
        if self._stop_event.is_set():
            await self._drain()
            self._stop_event.clear()

        context_id = str(uuid.uuid4())
        self._active_context_id = context_id
        self._generation += 1
        generation = self._generation

        try:
            await self._send_text_and_flush(text, context_id)
        except (websockets.exceptions.WebSocketException, OSError):
            logger.warning("Inworld TTS websocket dropped; reconnecting")
            await self._reset_connection()
            self._active_context_id = context_id
            await self._send_text_and_flush(text, context_id)

        return self._receive_audio(context_id, generation)

    async def stop_audio(self) -> None:
        """Stop current synthesis by closing the active context."""
        self._generation += 1
        self._stop_event.set()

        if self._active_context_id and self._websocket is not None:
            try:
                await self._send_close_context(self._active_context_id)
            except (websockets.exceptions.WebSocketException, OSError):
                await self._reset_connection()
            finally:
                self._active_context_id = None

    async def close(self) -> None:
        await self._reset_connection()
        await super().close()

    async def _send_text_and_flush(self, text: str, context_id: str) -> None:
        websocket = await self._ensure_connection()
        await self._send_create_context(context_id)
        await websocket.send(
            json.dumps(
                {
                    "send_text": {"text": text},
                    "contextId": context_id,
                }
            )
        )
        await websocket.send(
            json.dumps(
                {
                    "flush_context": {},
                    "contextId": context_id,
                }
            )
        )

    async def _receive_audio(
        self, context_id: str, generation: int
    ) -> AsyncIterator[PcmData]:
        websocket = await self._ensure_connection()
        should_close_context = True

        try:
            while True:
                if self._stop_event.is_set() or self._generation != generation:
                    break

                try:
                    message = await websocket.recv()
                except (websockets.exceptions.ConnectionClosed, OSError):
                    await self._reset_connection()
                    raise

                if not isinstance(message, str):
                    continue

                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    logger.warning("Skipping non-JSON Inworld TTS websocket message")
                    continue

                result = data.get("result", {})
                status = result.get("status", {})
                if status.get("code", 0) != 0:
                    error_message = status.get("message", "Unknown Inworld error")
                    if "max contexts limit reached" in error_message.lower():
                        logger.warning(
                            "Inworld context limit reached; resetting websocket"
                        )
                        await self._reset_connection()
                    raise RuntimeError(f"Inworld TTS websocket error: {error_message}")

                if "error" in data:
                    raise RuntimeError(f"Inworld TTS websocket error: {data['error']}")

                msg_context_id = result.get("contextId") or result.get("context_id")
                if msg_context_id and msg_context_id != context_id:
                    continue

                audio_chunk = result.get("audioChunk", {})
                audio_b64 = audio_chunk.get("audioContent")
                if audio_b64:
                    decoded = base64.b64decode(audio_b64)
                    pcm = _decode_audio(decoded, self._sample_rate)
                    if pcm:
                        yield pcm

                if "contextClosed" in result:
                    should_close_context = False
                    break

                if "flushCompleted" in result:
                    break
        finally:
            if self._active_context_id == context_id:
                self._active_context_id = None
            if should_close_context:
                try:
                    await self._send_close_context(context_id)
                except (websockets.exceptions.WebSocketException, OSError):
                    await self._reset_connection()

    async def _send_create_context(self, context_id: str) -> None:
        websocket = await self._ensure_connection()
        audio_config: dict[str, object] = {
            "audioEncoding": "LINEAR16",
            "sampleRateHertz": self._sample_rate,
        }
        if self._speaking_rate is not None:
            audio_config["speakingRate"] = self._speaking_rate

        create_config: dict[str, object] = {
            "voiceId": self._voice_id,
            "modelId": self._model_id,
            "audioConfig": audio_config,
            "temperature": self._temperature,
            "autoMode": self._auto_mode,
            "timestampType": "TIMESTAMP_TYPE_UNSPECIFIED",
        }
        if self._apply_text_normalization is not None:
            create_config["applyTextNormalization"] = self._apply_text_normalization

        await websocket.send(
            json.dumps(
                {
                    "create": create_config,
                    "contextId": context_id,
                }
            )
        )

    async def _send_close_context(self, context_id: str) -> None:
        if self._websocket is None:
            return
        await self._websocket.send(
            json.dumps(
                {
                    "close_context": {},
                    "contextId": context_id,
                }
            )
        )

    async def _ensure_connection(self) -> websockets.ClientConnection:
        if (
            self._websocket is not None
            and self._websocket.state is websockets.State.OPEN
        ):
            return self._websocket

        if self._websocket is not None:
            try:
                await self._websocket.close()
            except (websockets.exceptions.WebSocketException, OSError):
                pass
            self._websocket = None

        if self._keepalive_task is not None and self._keepalive_task.done():
            self._keepalive_task = None

        request_id = str(uuid.uuid4())
        self._websocket = await websockets.connect(
            self._ws_url,
            additional_headers={
                "Authorization": f"Basic {self._api_key}",
                "X-User-Agent": USER_AGENT,
                "X-Request-Id": request_id,
            },
        )
        if self._keepalive_task is None:
            self._keepalive_task = asyncio.create_task(self._keepalive_loop())
        return self._websocket

    async def _reset_connection(self) -> None:
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass
            self._keepalive_task = None

        if self._websocket is not None:
            try:
                await self._websocket.close()
            finally:
                self._websocket = None

        self._active_context_id = None
        self._stop_event.clear()

    async def _drain(self) -> None:
        websocket = self._websocket
        if websocket is None:
            return

        deadline = asyncio.get_running_loop().time() + 0.5
        while asyncio.get_running_loop().time() < deadline:
            try:
                await asyncio.wait_for(websocket.recv(), timeout=0.05)
            except TimeoutError:
                break
            except (websockets.exceptions.ConnectionClosed, OSError):
                await self._reset_connection()
                break

    async def _keepalive_loop(self) -> None:
        while True:
            await asyncio.sleep(KEEPALIVE_INTERVAL_SECONDS)
            websocket = self._websocket
            if websocket is None:
                return
            if websocket.state is not websockets.State.OPEN:
                if self._websocket is websocket:
                    self._websocket = None
                return
            payload: dict[str, object] = {"send_text": {"text": ""}}
            if self._active_context_id:
                payload["contextId"] = self._active_context_id
            try:
                await websocket.send(json.dumps(payload))
            except (websockets.exceptions.WebSocketException, OSError):
                if self._websocket is websocket:
                    self._websocket = None
                try:
                    await websocket.close()
                except (websockets.exceptions.WebSocketException, OSError):
                    pass
                return
