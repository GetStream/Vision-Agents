import asyncio
import json
import logging
import os
import time
from typing import Any, Literal, Optional
from urllib.parse import urlencode

import websockets
from getstream.video.rtc.track_util import PcmData
from vision_agents.core import stt
from vision_agents.core.edge.types import Participant
from vision_agents.core.stt import TranscriptResponse
from vision_agents.core.utils.utils import cancel_and_wait

logger = logging.getLogger(__name__)

DEFAULT_WEBSOCKET_URL = "wss://api.cartesia.ai/stt/turns/websocket"
DEFAULT_CARTESIA_VERSION = "2026-03-01"
CONNECTION_READY_TIMEOUT_SECONDS = 10.0


class STT(stt.STT):
    """Speech-to-Text plugin backed by Cartesia Ink."""

    turn_detection: bool = True
    eager_turn_detection: bool = True

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "ink-2",
        sample_rate: Literal[8000, 16000, 22050, 24000, 44100, 48000] = 16000,
        encoding: Literal["pcm_s16le"] = "pcm_s16le",
        cartesia_version: str = DEFAULT_CARTESIA_VERSION,
        websocket_url: str = DEFAULT_WEBSOCKET_URL,
        audio_chunk_duration_ms: int = 100,
    ) -> None:
        """Create a new Cartesia STT instance.

        Args:
            api_key: Cartesia API key; falls back to ``CARTESIA_API_KEY``.
            model: Cartesia STT model to use. Defaults to ``ink-2``.
            sample_rate: PCM sample rate sent to Cartesia.
            encoding: Audio encoding sent to Cartesia.
            cartesia_version: Cartesia API version query parameter.
            websocket_url: WebSocket endpoint, mainly useful for tests.
            audio_chunk_duration_ms: Maximum duration per websocket audio frame.
        """
        super().__init__(provider_name="cartesia")

        resolved_api_key = api_key or os.getenv("CARTESIA_API_KEY")
        if not resolved_api_key:
            raise ValueError("CARTESIA_API_KEY env var or api_key parameter required")
        self.api_key = resolved_api_key

        self.model = model
        self.sample_rate = sample_rate
        self.encoding = encoding
        self.cartesia_version = cartesia_version
        self.websocket_url = websocket_url
        self.audio_chunk_duration_ms = audio_chunk_duration_ms

        self.connection: websockets.ClientConnection | None = None
        self._connection_ready = asyncio.Event()
        self._listen_task: asyncio.Task[Any] | None = None
        self._current_participant: Participant | None = None
        self._audio_start_time: float | None = None
        self._turn_in_progress = False
        self._connection_error: Exception | None = None

    async def start(self) -> None:
        """Open the Cartesia realtime STT websocket."""
        await super().start()
        self._connection_error = None

        try:
            url = self._build_websocket_url()
            self.connection = await asyncio.wait_for(
                websockets.connect(
                    url,
                    additional_headers={"X-API-Key": self.api_key},
                ),
                timeout=10.0,
            )

            self._listen_task = asyncio.create_task(self._listen())
            self._connection_ready.set()
            self._on_connected()
        except Exception:
            self.started = False
            self.connection = None
            self._connection_error = None
            self._connection_ready.clear()
            raise

    async def process_audio(
        self,
        pcm_data: PcmData,
        participant: Optional[Participant] = None,
    ) -> None:
        """Send PCM audio to Cartesia for realtime transcription."""
        if self.closed:
            logger.warning("Cartesia STT is closed, ignoring audio")
            return

        if self._connection_error is not None:
            raise RuntimeError(
                "Cartesia STT websocket connection failed; call start() to reconnect"
            ) from self._connection_error

        if not self.started:
            raise RuntimeError("Cartesia STT is not started; call start() first")

        try:
            await asyncio.wait_for(
                self._connection_ready.wait(),
                timeout=CONNECTION_READY_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                "Timed out waiting for Cartesia STT websocket connection"
            ) from exc

        connection = self.connection
        if connection is None or not self._connection_ready.is_set():
            raise RuntimeError(
                "Cartesia STT websocket connection is not ready; "
                "call start() to reconnect"
            )

        self._current_participant = participant
        if self._audio_start_time is None:
            self._audio_start_time = time.perf_counter()

        resampled_pcm = pcm_data.resample(self.sample_rate, 1)
        audio_bytes = resampled_pcm.samples.tobytes()
        bytes_per_sample = 2
        frame_size = max(
            bytes_per_sample,
            int(
                self.sample_rate
                * bytes_per_sample
                * self.audio_chunk_duration_ms
                / 1000
            ),
        )
        for offset in range(0, len(audio_bytes), frame_size):
            await connection.send(audio_bytes[offset : offset + frame_size])

    async def clear(self) -> None:
        self._turn_in_progress = False
        self._audio_start_time = None
        await super().clear()

    async def close(self) -> None:
        await super().close()

        if self._listen_task is not None:
            await cancel_and_wait(self._listen_task)
            self._listen_task = None

        if self.connection is not None:
            try:
                await self.connection.close()
            except Exception as exc:
                logger.warning("Error closing Cartesia STT websocket: %s", exc)
            finally:
                self.connection = None
                self._connection_ready.clear()
                self._on_disconnected(clean=True)

    def _build_websocket_url(self) -> str:
        query = urlencode(
            {
                "model": self.model,
                "encoding": self.encoding,
                "sample_rate": str(self.sample_rate),
                "cartesia_version": self.cartesia_version,
            }
        )
        separator = "&" if "?" in self.websocket_url else "?"
        return f"{self.websocket_url}{separator}{query}"

    async def _listen(self) -> None:
        assert self.connection is not None
        try:
            async for message in self.connection:
                self._handle_message(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self.closed:
                logger.exception("Cartesia STT websocket error")
                self._emit_error_event(exc, context="listen")
                self._connection_error = exc
                self.started = False
                self.connection = None
                self._connection_ready.clear()
                self._on_disconnected(reason=str(exc), clean=False)

    def _handle_message(self, message: str | bytes | dict[str, Any]) -> None:
        if isinstance(message, bytes):
            message = message.decode("utf-8")
        if isinstance(message, str):
            data = json.loads(message)
        else:
            data = message

        event_type = data.get("type")
        if event_type in {"turn.start", "turn_start"}:
            self._handle_turn_started(data)
        elif event_type in {"turn.update", "turn_update"}:
            self._handle_transcript(data, final=False)
        elif event_type in {"turn.eager_end", "turn_eager_end"}:
            self._handle_transcript(data, final=False)
            self._handle_turn_ended(data, eager=True)
        elif event_type in {"turn.resume", "turn_resume"}:
            self._handle_turn_resumed(data)
        elif event_type in {"turn.end", "turn_end"}:
            self._handle_transcript(data, final=True)
            self._handle_turn_ended(data, eager=False)
        elif event_type == "error" or data.get("error"):
            error = RuntimeError(str(data.get("error") or data))
            self._emit_error_event(error, context="message")
        else:
            logger.debug("Unhandled Cartesia STT event: %s", event_type)

    def _handle_turn_started(self, data: dict[str, Any]) -> None:
        participant = self._current_participant
        if participant is None:
            logger.warning("Received Cartesia turn start but no participant set")
            return

        self._turn_in_progress = True
        self._emit_turn_started_event(
            participant,
            confidence=self._confidence(data),
        )

    def _handle_turn_resumed(self, data: dict[str, Any]) -> None:
        participant = self._current_participant
        if participant is None:
            logger.warning("Received Cartesia turn resume but no participant set")
            return

        self._turn_in_progress = True
        self._emit_turn_started_event(
            participant,
            confidence=self._confidence(data),
        )

    def _handle_turn_ended(self, data: dict[str, Any], *, eager: bool) -> None:
        participant = self._current_participant
        if participant is None:
            logger.warning("Received Cartesia turn end but no participant set")
            return

        if not eager:
            self._turn_in_progress = False
            self._audio_start_time = None

        self._emit_turn_ended_event(
            participant,
            confidence=self._confidence(data),
            eager=eager,
            duration_ms=self._duration_ms(data),
            trailing_silence_ms=self._trailing_silence_ms(data),
        )

    def _handle_transcript(self, data: dict[str, Any], *, final: bool) -> None:
        transcript_text = self._transcript_text(data)
        if not transcript_text:
            return

        participant = self._current_participant
        if participant is None:
            logger.warning("Received Cartesia transcript but no participant set")
            return

        processing_time_ms: float | None = None
        if self._audio_start_time is not None:
            processing_time_ms = (time.perf_counter() - self._audio_start_time) * 1000

        response = TranscriptResponse(
            confidence=self._confidence(data),
            language=data.get("language"),
            audio_duration_ms=self._duration_ms(data),
            model_name=self.model,
            processing_time_ms=processing_time_ms,
            other={"words": data.get("words")} if data.get("words") else None,
        )
        self._emit_transcript_event(
            transcript_text,
            participant,
            response,
            mode="final" if final else "replacement",
        )

    @staticmethod
    def _transcript_text(data: dict[str, Any]) -> str:
        transcript = data.get("transcript")
        if transcript is None:
            transcript = data.get("text")
        return str(transcript or "")

    @staticmethod
    def _confidence(data: dict[str, Any]) -> float:
        confidence = data.get("confidence")
        if isinstance(confidence, int | float):
            return float(confidence)

        words = data.get("words") or []
        confidences = [
            float(word[key])
            for word in words
            for key in ("confidence", "score")
            if isinstance(word, dict) and isinstance(word.get(key), int | float)
        ]
        if confidences:
            return sum(confidences) / len(confidences)
        return 0.0

    @staticmethod
    def _duration_ms(data: dict[str, Any]) -> float | None:
        if isinstance(data.get("duration_ms"), int | float):
            return float(data["duration_ms"])
        if isinstance(data.get("audio_duration_ms"), int | float):
            return float(data["audio_duration_ms"])
        if isinstance(data.get("duration"), int | float):
            return float(data["duration"]) * 1000
        return None

    @staticmethod
    def _trailing_silence_ms(data: dict[str, Any]) -> float | None:
        if isinstance(data.get("trailing_silence_ms"), int | float):
            return float(data["trailing_silence_ms"])
        if isinstance(data.get("trailing_silence"), int | float):
            return float(data["trailing_silence"]) * 1000
        return None
