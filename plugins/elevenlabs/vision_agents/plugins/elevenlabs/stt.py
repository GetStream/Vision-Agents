import asyncio
import base64
import logging
import os
import time
from typing import Any, Optional

from elevenlabs import (
    AudioFormat,
    CommitStrategy,
    RealtimeAudioOptions,
    RealtimeConnection,
    RealtimeEvents,
)
from elevenlabs.client import AsyncElevenLabs
from getstream.video.rtc.track_util import PcmData
from vision_agents.core import stt
from vision_agents.core.edge.types import Participant
from vision_agents.core.stt import TranscriptResponse
from vision_agents.core.utils.audio_queue import AudioQueue
from vision_agents.core.utils.utils import cancel_and_wait

logger = logging.getLogger(__name__)


SAMPLE_RATE = 16000


class ElevenlabsSTT(stt.STT):
    """
    ElevenLabs Scribe v2 Realtime Speech-to-Text implementation.

    Features:
    - Real-time speech recognition with low latency (~150ms)
    - 99 language support
    - VAD-based commit strategy
    - Automatic reconnection with exponential backoff

    Docs:
    - https://elevenlabs.io/docs/models#scribe-v2-realtime
    """

    turn_detection: bool = True
    connection: Optional[RealtimeConnection] = None

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = "scribe_v2_realtime",
        language_code: str = "en",
        vad_silence_threshold_secs: float = 0.3,
        vad_threshold: float = 0.4,
        min_speech_duration_ms: int = 100,
        min_silence_duration_ms: int = 100,
        audio_chunk_duration_ms: int = 100,
        keepalive_interval_ms: int = 5000,
        client: Optional[AsyncElevenLabs] = None,
    ):
        """
        Initialize ElevenLabs Scribe v2 STT.

        Args:
            api_key: ElevenLabs API key. If not provided, ELEVENLABS_API_KEY env var is used.
            model_id: Model to use for transcription. Defaults to "scribe_v2_realtime".
            language_code: Language code (e.g., "en", "es"). Defaults to "en".
            vad_silence_threshold_secs: VAD silence threshold in seconds.
            vad_threshold: VAD threshold for speech detection.
            min_speech_duration_ms: Minimum speech duration in milliseconds.
            min_silence_duration_ms: Minimum silence duration in milliseconds.
            audio_chunk_duration_ms: Duration of audio chunks to send (100-1000ms recommended).
            keepalive_interval_ms: Send a silence frame after this many ms of idle
                so the server doesn't close the WS (e.g. while the user is muted).
            client: Optional pre-configured AsyncElevenLabs instance.
        """
        super().__init__(provider_name="elevenlabs")

        if not api_key:
            api_key = os.environ.get("ELEVENLABS_API_KEY")

        self.client = client if client is not None else AsyncElevenLabs(api_key=api_key)

        self.model_id = model_id
        self.language_code = language_code
        self.vad_silence_threshold_secs = vad_silence_threshold_secs
        self.vad_threshold = vad_threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.audio_chunk_duration_ms = audio_chunk_duration_ms
        self.keepalive_interval_ms = keepalive_interval_ms

        # Pre-generated 2s of silence for keep-alive purposes.
        silence_bytes = b"\x00" * (SAMPLE_RATE * audio_chunk_duration_ms // 1000 * 2)
        self._silence_2s_b64 = base64.b64encode(silence_bytes).decode("utf-8")

        self._current_participant: Optional[Participant] = None
        self._connection_ready = asyncio.Event()
        self._listen_task: Optional[asyncio.Task[Any]] = None
        self._send_task: Optional[asyncio.Task[Any]] = None
        self._audio_queue: Optional[AudioQueue] = None
        self._should_reconnect = {"value": False}
        self._reconnect_event = asyncio.Event()
        self._turn_in_progress = False
        # Track when audio processing started for latency measurement
        self._audio_start_time: Optional[float] = None

    async def process_audio(
        self,
        pcm_data: PcmData,
        participant: Optional[Participant] = None,
    ):
        """
        Process audio data through ElevenLabs Scribe v2 for transcription.

        This method adds audio to a queue which is then sent to the WebSocket connection
        in batches. Audio is automatically resampled to 16kHz mono for optimal quality.

        Args:
            pcm_data: The PCM audio data to process.
            participant: Optional participant metadata.
        """
        if self.closed:
            logger.warning("ElevenLabs STT is closed, ignoring audio")
            return

        # Wait for connection to be ready
        await self._connection_ready.wait()

        # Double-check connection is still ready
        if not self._connection_ready.is_set():
            logger.warning("ElevenLabs connection closed while processing audio")
            return

        # Resample to 16kHz mono (recommended by ElevenLabs)
        resampled_pcm = pcm_data.resample(16_000, 1)

        self._current_participant = participant

        # Track start time for first audio chunk of a new utterance
        if self._audio_start_time is None:
            self._audio_start_time = time.perf_counter()

        # Add to audio queue for batching
        if self._audio_queue is not None:
            await self._audio_queue.put(resampled_pcm)

    async def start(self):
        """
        Start the ElevenLabs WebSocket connection and begin listening for transcripts.
        """
        if self.connection is not None:
            logger.warning("ElevenLabs connection already started")
            return

        # Initialize audio queue with 10 second buffer limit
        self._audio_queue = AudioQueue(buffer_limit_ms=10000)

        # Configure realtime audio options (TypedDict)
        audio_options: RealtimeAudioOptions = {
            "model_id": self.model_id,
            "language_code": self.language_code,
            "audio_format": AudioFormat.PCM_16000,
            "sample_rate": SAMPLE_RATE,
            "commit_strategy": CommitStrategy.VAD,
            "vad_silence_threshold_secs": self.vad_silence_threshold_secs,
            "vad_threshold": self.vad_threshold,
            "min_speech_duration_ms": self.min_speech_duration_ms,
            "min_silence_duration_ms": self.min_silence_duration_ms,
            "include_timestamps": True,
        }

        # Connect to ElevenLabs realtime speech-to-text
        self.connection = await asyncio.wait_for(
            self.client.speech_to_text.realtime.connect(audio_options), timeout=10.0
        )

        # Register event handlers
        if self.connection is not None:
            self.connection.on(
                RealtimeEvents.PARTIAL_TRANSCRIPT, self._on_partial_transcript
            )
            self.connection.on(
                RealtimeEvents.COMMITTED_TRANSCRIPT_WITH_TIMESTAMPS,
                self._on_committed_transcript,
            )
            self.connection.on(RealtimeEvents.ERROR, self._on_error)
            self.connection.on(RealtimeEvents.CLOSE, self._on_close)

        logger.info("ElevenLabs WebSocket connection established")

        # Start the audio sending task
        self._send_task = asyncio.create_task(self._send_audio_loop())

        # Mark connection as ready
        self._connection_ready.set()

    async def _send_audio_loop(self):
        """Send audio chunks from the queue, falling back to silence on idle."""
        keepalive_s = self.keepalive_interval_ms / 1000
        last_send_at = time.monotonic()

        while (
            not self.closed
            and self.connection is not None
            and self._audio_queue is not None
        ):
            try:
                pcm_chunk = await asyncio.wait_for(
                    self._audio_queue.get_duration(self.audio_chunk_duration_ms),
                    timeout=keepalive_s,
                )
                audio_bytes = pcm_chunk.samples.tobytes()
                audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
            except (asyncio.TimeoutError, asyncio.QueueEmpty):
                # get_samples raises QueueEmpty after its own 100ms timeout,
                # which bypasses wait_for — so gate the silence send on
                # wall-clock time since the last actual send.
                if time.monotonic() - last_send_at < keepalive_s:
                    continue
                audio_base64 = self._silence_2s_b64

            try:
                await self.connection.send(
                    {
                        "audio_base_64": audio_base64,
                        "sample_rate": SAMPLE_RATE,
                    }
                )
                last_send_at = time.monotonic()
            except Exception:
                if not self.closed:
                    logger.exception("Error sending audio to ElevenLabs")
                break

    def _on_partial_transcript(self, transcription_data: dict[str, Any]):
        """
        Event handler for partial transcription results from ElevenLabs.

        Args:
            transcription_data: The partial transcription result from ElevenLabs (dict)
        """
        # Extract transcript text from dict
        if isinstance(transcription_data, dict):
            transcript_text = transcription_data.get("text", "").strip()
            confidence = transcription_data.get("confidence", 0.0)
            words = transcription_data.get("words")
        else:
            raise Exception(
                "unexpected type for transcription data. expected dict got {}".format(
                    type(transcription_data)
                )
            )

        if not transcript_text:
            return

        # Build response metadata with word timestamps if available
        other: Optional[dict[str, Any]] = None
        if words:
            other = {"words": words}

        # Calculate processing time (time from first audio to transcript)
        processing_time_ms: Optional[float] = None
        if self._audio_start_time is not None:
            processing_time_ms = (time.perf_counter() - self._audio_start_time) * 1000

        response_metadata = TranscriptResponse(
            confidence=confidence,
            language=self.language_code,
            model_name=self.model_id,
            other=other,
            processing_time_ms=processing_time_ms,
        )

        # Use the participant from the most recent process_audio call
        participant = self._current_participant

        if participant is None:
            raise ValueError(
                "No participant set - audio must be processed with a participant"
            )

        # Signal turn start on the first partial of a new utterance
        if not self._turn_in_progress:
            self._turn_in_progress = True
            self._emit_turn_started_event(participant)

        # Emit partial transcript
        self._emit_transcript_event(
            transcript_text, participant, response_metadata, mode="replacement"
        )

    def _on_committed_transcript(self, transcription_data: dict[str, Any]):
        """Event handler for final (committed) transcription results from ElevenLabs."""
        if not isinstance(transcription_data, dict):
            raise TypeError(
                f"Unexpected type for transcription data; "
                f"expected dict, got {type(transcription_data)}"
            )

        # Empty text can come from the server committing our keep-alive silence
        # (no turn in progress), or from a real end-of-segment where VAD closed
        # the segment with no transcribable audio.
        transcript_text = transcription_data.get("text", "").strip()
        if not transcript_text and not self._turn_in_progress:
            return

        participant = self._current_participant
        if participant is None:
            raise ValueError(
                "No participant set - audio must be processed with a participant"
            )

        if not transcript_text:
            # _turn_in_progress is True (we returned earlier otherwise).
            self._turn_in_progress = False
            self._audio_start_time = None
            self._emit_turn_ended_event(participant)
            return

        processing_time_ms: Optional[float] = None
        if self._audio_start_time is not None:
            processing_time_ms = (time.perf_counter() - self._audio_start_time) * 1000

        words = transcription_data.get("words")
        response_metadata = TranscriptResponse(
            confidence=transcription_data.get("confidence", 0.0),
            language=self.language_code,
            model_name=self.model_id,
            other={"words": words} if words else None,
            processing_time_ms=processing_time_ms,
        )

        self._emit_transcript_event(
            transcript_text, participant, response_metadata, mode="final"
        )
        self._audio_start_time = None

        # Gated so a second commit for the same utterance does not double-fire.
        if self._turn_in_progress:
            self._turn_in_progress = False
            self._emit_turn_ended_event(participant)

    def _on_error(self, error: Any):
        """
        Event handler for errors from ElevenLabs.

        Args:
            error: The error from ElevenLabs
        """
        if self.closed:
            logger.debug(f"ElevenLabs WebSocket error during shutdown: {error}")
            self._turn_in_progress = False
            self._current_participant = None
            return
        logger.error(f"ElevenLabs WebSocket error: {error}")
        # Reset audio start time to avoid incorrect metrics on next utterance
        self._audio_start_time = None
        self._turn_in_progress = False
        self._current_participant = None
        self._should_reconnect["value"] = True
        self._reconnect_event.set()

    def _on_close(self):
        """
        Event handler for connection close.
        """
        if not self.closed:
            logger.warning("ElevenLabs WebSocket connection closed")
        # Reset audio start time to avoid incorrect metrics on next utterance
        self._audio_start_time = None
        self._turn_in_progress = False
        self._current_participant = None
        self._connection_ready.clear()
        self._reconnect_event.set()

    async def _attempt_reconnect(self):
        """
        Attempt to reconnect to ElevenLabs with exponential backoff.
        """
        if not self._should_reconnect["value"]:
            return

        logger.info("Attempting to reconnect to ElevenLabs...")

        for attempt in range(3):
            try:
                await asyncio.sleep(2**attempt)  # Exponential backoff

                # Clear the old connection
                self.connection = None
                self._connection_ready.clear()

                # Attempt to reconnect
                await self.start()

                logger.info(f"Reconnection successful on attempt {attempt + 1}")
                self._should_reconnect["value"] = False
                return

            except Exception as e:
                logger.error(f"Reconnection attempt {attempt + 1} failed: {e}")

        logger.error("Failed to reconnect after 3 attempts")

    async def clear(self):
        """No-op: VAD commit strategy handles commits automatically."""
        self._audio_start_time = None
        self._turn_in_progress = False
        self._current_participant = None
        if self._audio_queue is not None:
            self._audio_queue.clear()

    async def close(self):
        """Close the ElevenLabs connection and clean up resources."""
        # Mark as closed first
        await super().close()

        # Reset audio start time to avoid incorrect metrics if reused
        self._audio_start_time = None

        # Cancel send task
        if self._send_task:
            await cancel_and_wait(self._send_task)

        # Cancel listen task
        if self._listen_task:
            await cancel_and_wait(self._listen_task)

        # Close connection
        if self.connection:
            try:
                await self.connection.close()
            except Exception as e:
                logger.warning(f"Error closing Elevenlabs connection: {e}")
            finally:
                self.connection = None
                self._connection_ready.clear()
