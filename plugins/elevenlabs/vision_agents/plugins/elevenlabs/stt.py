import asyncio
import base64
import logging
import os
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


class STT(stt.STT):
    """
    ElevenLabs Scribe v2 Realtime Speech-to-Text implementation.

    Features:
    - Real-time speech recognition with low latency (~150ms)
    - 99 language support
    - VAD-based commit strategy
    - Automatic reconnection with exponential backoff

    Docs:
    - https://elevenlabs.io/docs/models#scribe-v2-realtime

    Note: This model does NOT support turn detection.
    """

    turn_detection: bool = False  # Scribe v2 does not support turn detection
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

        self._current_participant: Optional[Participant] = None
        self._connection_ready = asyncio.Event()
        self._listen_task: Optional[asyncio.Task[Any]] = None
        self._send_task: Optional[asyncio.Task[Any]] = None
        self._audio_queue: Optional[AudioQueue] = None
        self._should_reconnect = {"value": False}
        self._reconnect_event = asyncio.Event()
        self._commit_received = asyncio.Event()

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

        # Initialize audio queue with 5 second buffer limit
        self._audio_queue = AudioQueue(buffer_limit_ms=10000)

        # Configure realtime audio options (TypedDict)
        audio_options: RealtimeAudioOptions = {
            "model_id": self.model_id,
            "language_code": self.language_code,
            "audio_format": AudioFormat.PCM_16000,
            "sample_rate": 16000,
            "commit_strategy": CommitStrategy.MANUAL,  # manual works best
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
        """
        Continuously send audio chunks from the queue to the WebSocket connection.
        Batches audio into chunks of the configured duration.
        """
        try:
            while not self.closed and self.connection is not None:
                try:
                    # Get audio chunk of configured duration from queue
                    if self._audio_queue is not None:
                        pcm_chunk = await self._audio_queue.get_duration(
                            self.audio_chunk_duration_ms
                        )

                        # Convert int16 samples to bytes
                        audio_bytes = pcm_chunk.samples.tobytes()

                        # Encode to base64
                        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

                        # Send to ElevenLabs
                        await self.connection.send(
                            {
                                "audio_base_64": audio_base64,
                                "sample_rate": 16000,
                            }
                        )

                except asyncio.QueueEmpty:
                    # No audio available, wait a bit
                    await asyncio.sleep(0.01)
                except Exception as e:
                    if not self.closed:
                        logger.error(f"Error sending audio to ElevenLabs: {e}")
                    break

        except asyncio.CancelledError:
            logger.debug("Audio send loop cancelled")
            raise

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

        response_metadata = TranscriptResponse(
            confidence=confidence,
            language=self.language_code,
            model_name=self.model_id,
            other=other,
        )

        # Use the participant from the most recent process_audio call
        participant = self._current_participant

        if participant is None:
            raise ValueError(
                "No participant set - audio must be processed with a participant"
            )

        # Emit partial transcript
        self._emit_partial_transcript_event(
            transcript_text, participant, response_metadata
        )

    def _on_committed_transcript(self, transcription_data: dict[str, Any]):
        """
        Event handler for final (committed) transcription results from ElevenLabs.

        Args:
            transcription_data: The committed transcription result from ElevenLabs (dict)
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

        response_metadata = TranscriptResponse(
            confidence=confidence,
            language=self.language_code,
            model_name=self.model_id,
            other=other,
        )

        # Use the participant from the most recent process_audio call
        participant = self._current_participant

        if participant is None:
            raise ValueError(
                "No participant set - audio must be processed with a participant"
            )

        # Emit final transcript
        self._emit_transcript_event(transcript_text, participant, response_metadata)

        # Signal that commit was received
        self._commit_received.set()

    def _on_error(self, error: Any):
        """
        Event handler for errors from ElevenLabs.

        Args:
            error: The error from ElevenLabs
        """
        logger.error(f"ElevenLabs WebSocket error: {error}")
        self._should_reconnect["value"] = True
        self._reconnect_event.set()

    def _on_close(self):
        """
        Event handler for connection close.
        """
        logger.warning("ElevenLabs WebSocket connection closed")
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

    async def clear(self, timeout: float = 10.0):
        """
        Commit any pending audio and wait for the final transcript.

        Args:
            timeout: Maximum time to wait for the committed transcript in seconds.
        """
        if not self.connection:
            return

        # Clear the event before committing
        self._commit_received.clear()

        await self.connection.commit()

        # Wait for the committed transcript event
        try:
            await asyncio.wait_for(self._commit_received.wait(), timeout=timeout)
        except TimeoutError:
            logger.warning("Timeout waiting for committed transcript after clear()")

    async def close(self):
        """Close the ElevenLabs connection and clean up resources."""
        # Mark as closed first
        await super().close()

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
