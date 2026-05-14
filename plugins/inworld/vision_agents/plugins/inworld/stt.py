import asyncio
import base64
import json
import logging
import os
import time
import uuid

import websockets
import websockets.exceptions
from getstream.video.rtc.track_util import PcmData
from vision_agents.core import stt
from vision_agents.core.edge.types import Participant
from vision_agents.core.stt import TranscriptResponse
from vision_agents.core.utils.utils import cancel_and_wait

logger = logging.getLogger(__name__)

INWORLD_STT_WS_URL = "wss://api.inworld.ai/stt/v1/transcribe:streamBidirectional"
DEFAULT_MODEL = "inworld/inworld-stt-1"
DEFAULT_LANGUAGE = "en-US"
DEFAULT_SAMPLE_RATE = 16000


class STT(stt.STT):
    """Inworld AI Speech-to-Text using bidirectional WebSocket streaming."""

    turn_detection: bool = True

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str = DEFAULT_MODEL,
        language: str = DEFAULT_LANGUAGE,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        num_channels: int = 1,
        end_of_turn_confidence_threshold: float = 0.5,
        min_end_of_turn_silence_when_confident: int | None = None,
        vad_threshold: float | None = None,
        ws_url: str = INWORLD_STT_WS_URL,
        max_reconnect_attempts: int = 3,
        reconnect_backoff_initial_s: float = 0.5,
        reconnect_backoff_max_s: float = 4.0,
    ):
        """Initialize the Inworld STT service.

        Args:
            api_key: Inworld API key. Falls back to INWORLD_API_KEY env var.
            model_id: STT model identifier (e.g. "inworld/inworld-stt-1").
            language: BCP-47 language code (e.g. "en-US").
            sample_rate: Audio sample rate in Hz.
            num_channels: Number of audio channels.
            end_of_turn_confidence_threshold: Confidence threshold for end-of-turn.
                Range [0.0, 1.0]. Higher values reduce false positives.
            min_end_of_turn_silence_when_confident: Min silence (ms) before end-of-turn
                when confidence is high.
            vad_threshold: Voice activity detection threshold. Range [0.0, 1.0].
            ws_url: WebSocket endpoint URL.
            max_reconnect_attempts: Max reconnect attempts on failure.
            reconnect_backoff_initial_s: Initial backoff delay in seconds.
            reconnect_backoff_max_s: Maximum backoff delay in seconds.
        """
        super().__init__(provider_name="inworld")

        api_key = api_key or os.getenv("INWORLD_API_KEY")
        if not api_key:
            raise ValueError(
                "INWORLD_API_KEY environment variable must be set or api_key must be provided"
            )

        self._api_key = api_key
        self._model_id = model_id
        self._language = language
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._end_of_turn_confidence_threshold = end_of_turn_confidence_threshold
        self._min_end_of_turn_silence_when_confident = (
            min_end_of_turn_silence_when_confident
        )
        self._vad_threshold = vad_threshold
        self._ws_url = ws_url

        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_backoff_initial_s = reconnect_backoff_initial_s
        self._reconnect_backoff_max_s = reconnect_backoff_max_s

        self._websocket: websockets.ClientConnection | None = None
        self._receive_task: asyncio.Task | None = None
        self._send_task: asyncio.Task | None = None
        self._audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._connection_ready = asyncio.Event()
        self._current_participant: Participant | None = None
        self._audio_start_time: float | None = None
        self._speaking = False
        # Buffer to 100ms chunks (int16 = 2 bytes per sample)
        self._chunk_size = self._sample_rate * 2 // 10
        self._audio_buffer = bytearray()

    def _build_transcribe_config(self) -> dict:
        config: dict = {
            "modelId": self._model_id,
            "audioEncoding": "LINEAR16",
            "sampleRateHertz": self._sample_rate,
            "numberOfChannels": self._num_channels,
            "language": self._language,
            "endOfTurnConfidenceThreshold": self._end_of_turn_confidence_threshold,
        }

        inworld_v1_config: dict = {}
        if self._min_end_of_turn_silence_when_confident is not None:
            inworld_v1_config["minEndOfTurnSilenceWhenConfident"] = (
                self._min_end_of_turn_silence_when_confident
            )
        if self._vad_threshold is not None:
            inworld_v1_config["vadThreshold"] = self._vad_threshold
        if inworld_v1_config:
            config["inworldSttV1Config"] = inworld_v1_config

        return config

    async def start(self) -> None:
        """Open the Inworld STT WebSocket connection and begin listening."""
        await super().start()
        await self._connect()
        try:
            await asyncio.wait_for(self._connection_ready.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            raise TimeoutError(
                "Failed to connect to Inworld STT within 10 seconds"
            ) from None

    async def _connect(self) -> None:
        request_id = str(uuid.uuid4())
        self._websocket = await websockets.connect(
            self._ws_url,
            additional_headers={
                "Authorization": f"Basic {self._api_key}",
                "X-Request-Id": request_id,
            },
        )
        config_msg = json.dumps({"transcribeConfig": self._build_transcribe_config()})
        await self._websocket.send(config_msg)
        self._connection_ready.set()
        self._receive_task = asyncio.create_task(self._receive_loop())
        self._send_task = asyncio.create_task(self._send_loop())

    async def _disconnect(self) -> None:
        current = asyncio.current_task()
        tasks = [
            t
            for t in (self._send_task, self._receive_task)
            if t is not None and t is not current
        ]
        if tasks:
            await cancel_and_wait(*tasks)
        self._send_task = None
        self._receive_task = None

        if self._websocket is not None:
            try:
                await self._websocket.close()
            except (websockets.exceptions.WebSocketException, OSError):
                pass
        self._websocket = None

    async def _reconnect(self) -> bool:
        """Attempt bounded reconnect with exponential backoff.

        Returns:
            True if reconnect succeeded, False if attempts exhausted.
        """
        await self._disconnect()
        self._connection_ready.clear()
        self._speaking = False

        delay = self._reconnect_backoff_initial_s
        for attempt in range(1, self._max_reconnect_attempts + 1):
            if self.closed:
                return False
            logger.info(
                "Inworld STT reconnect attempt %d/%d in %.1fs",
                attempt,
                self._max_reconnect_attempts,
                delay,
            )
            await asyncio.sleep(delay)
            try:
                await self._connect()
                await asyncio.wait_for(self._connection_ready.wait(), timeout=10.0)
                logger.info("Inworld STT reconnected on attempt %d", attempt)
                return True
            except (
                websockets.exceptions.WebSocketException,
                asyncio.TimeoutError,
                OSError,
            ):
                logger.exception("Inworld STT reconnect attempt %d failed", attempt)
                await self._disconnect()
            delay = min(delay * 2, self._reconnect_backoff_max_s)

        logger.error(
            "Inworld STT reconnect failed after %d attempts",
            self._max_reconnect_attempts,
        )
        return False

    async def _receive_loop(self) -> None:
        if self._websocket is None:
            return
        try:
            async for message in self._websocket:
                if not isinstance(message, str):
                    continue
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue
                self._handle_message(data)
        except asyncio.CancelledError:
            raise
        except websockets.exceptions.ConnectionClosed as exc:
            logger.warning(
                "Inworld STT WebSocket closed: code=%s reason=%s",
                exc.code,
                exc.reason,
            )
        except OSError:
            logger.warning("Inworld STT WebSocket connection lost")
        except Exception:
            logger.exception("Error in Inworld STT receive loop")

        if not self.closed:
            reconnected = await self._reconnect()
            if not reconnected:
                self.closed = True

    async def _send_loop(self) -> None:
        try:
            while True:
                chunk = await self._audio_queue.get()
                if chunk is None:
                    break
                if self._websocket is not None:
                    audio_b64 = base64.b64encode(chunk).decode()
                    await self._websocket.send(
                        json.dumps({"audioChunk": {"content": audio_b64}})
                    )
        except asyncio.CancelledError:
            raise
        except (websockets.exceptions.WebSocketException, OSError):
            logger.warning("Inworld STT send loop disconnected")
        except Exception:
            logger.exception("Error in Inworld STT send loop")

    def _handle_message(self, data: dict) -> None:
        if "error" in data:
            logger.error("Inworld STT server error: %s", data["error"])
            return

        result = data.get("result", {})

        if "speechStarted" in result:
            self._speaking = True
            participant = self._current_participant
            if participant is not None:
                confidence = result["speechStarted"].get("confidence", 0.5)
                self._emit_turn_started_event(participant, confidence=confidence)
            return

        if "speechStopped" in result:
            silence_ms = result["speechStopped"].get("silenceDurationMs")
            if self._speaking:
                self._speaking = False
                self._audio_start_time = None
                participant = self._current_participant
                if participant is not None:
                    self._emit_turn_ended_event(
                        participant,
                        trailing_silence_ms=float(silence_ms)
                        if silence_ms is not None
                        else None,
                    )
            return

        transcription = result.get("transcription", {})
        if not transcription:
            return

        text = transcription.get("transcript", "")
        is_final = transcription.get("isFinal", False)

        if not text and not is_final:
            return

        participant = self._current_participant
        if participant is None:
            logger.warning("Received transcript but no participant available")
            return

        processing_time_ms: float | None = None
        if self._audio_start_time is not None:
            processing_time_ms = (time.perf_counter() - self._audio_start_time) * 1000

        response = TranscriptResponse(
            model_name=self._model_id,
            language=self._language,
            processing_time_ms=processing_time_ms,
        )

        if text:
            if is_final:
                self._emit_transcript_event(text, participant, response, mode="final")
                self._audio_start_time = None
            else:
                self._emit_transcript_event(
                    text, participant, response, mode="replacement"
                )

        if is_final and self._speaking:
            self._speaking = False
            self._emit_turn_ended_event(participant)

    async def process_audio(
        self,
        pcm_data: PcmData,
        participant: Participant | None = None,
    ) -> None:
        """Process audio data through Inworld for transcription.

        Args:
            pcm_data: The PCM audio data to process.
            participant: Optional participant metadata.
        """
        if self.closed:
            logger.warning("Inworld STT is closed, ignoring audio")
            return

        await self._connection_ready.wait()

        resampled = pcm_data.resample(self._sample_rate, self._num_channels)
        audio_bytes = resampled.samples.tobytes()

        self._current_participant = participant

        if self._audio_start_time is None:
            self._audio_start_time = time.perf_counter()

        self._audio_buffer.extend(audio_bytes)
        while len(self._audio_buffer) >= self._chunk_size:
            chunk = bytes(self._audio_buffer[: self._chunk_size])
            del self._audio_buffer[: self._chunk_size]
            await self._audio_queue.put(chunk)

    async def clear(self) -> None:
        """Reset turn state on barge-in or interruption."""
        await super().clear()
        self._speaking = False
        self._audio_start_time = None
        self._audio_buffer.clear()

    async def close(self) -> None:
        """Close the Inworld STT WebSocket connection and clean up resources."""
        await super().close()

        if self._audio_buffer:
            await self._audio_queue.put(bytes(self._audio_buffer))
            self._audio_buffer.clear()

        await self._audio_queue.put(None)
        if self._send_task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._send_task), timeout=2.0)
            except asyncio.TimeoutError:
                pass  # _disconnect will cancel and await it

        if self._websocket is not None:
            try:
                await self._websocket.send(json.dumps({"closeStream": {}}))
            except (websockets.exceptions.WebSocketException, OSError):
                logger.debug("Could not send closeStream message")

        await self._disconnect()
        self._connection_ready.clear()
        self._audio_start_time = None
        self._speaking = False
