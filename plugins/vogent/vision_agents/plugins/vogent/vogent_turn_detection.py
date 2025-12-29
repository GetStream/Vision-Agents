import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Literal, Optional

from faster_whisper import WhisperModel
from getstream.video.rtc.track_util import AudioFormat, PcmData
from vision_agents.core.agents import Conversation
from vision_agents.core.edge.types import Participant
from vision_agents.core.turn_detection import (
    TurnDetector,
    TurnStartedEvent,
)
from vision_agents.core.vad.silero import SileroVADSession, SileroVADSessionPool
from vision_agents.core.warmup import Warmable
from vogent_turn import TurnDetector as VogentDetector

logger = logging.getLogger(__name__)

# Audio processing constants
CHUNK = 512  # Samples per chunk for VAD processing
RATE = 16000  # Sample rate in Hz (16kHz)


@dataclass
class Silence:
    trailing_silence_chunks: int = 0
    speaking_chunks: int = 0


class VogentTurnDetection(
    TurnDetector, Warmable[tuple[SileroVADSessionPool, WhisperModel, VogentDetector]]
):
    """
    Vogent Turn Detection combines audio intonation and text context for accurate turn detection.

    This implementation:
    1. Uses Silero VAD to detect when speech starts/stops
    2. Uses faster-whisper to transcribe audio in real-time
    3. Uses Vogent Turn model (multimodal) to detect turn completion

    Vogent operates on both audio features AND text context, making it more accurate
    than audio-only approaches, especially for handling incomplete thoughts.

    Reference: https://github.com/vogent/vogent-turn
    Blogpost: https://blog.vogent.ai/posts/voturn-80m-state-of-the-art-turn-detection-for-voice-agents
    """

    def __init__(
        self,
        whisper_model_size: Literal[
            "tiny", "base", "small", "medium", "large"
        ] = "tiny",
        vad_reset_interval_seconds: float = 5.0,
        speech_probability_threshold: float = 0.5,
        pre_speech_buffer_ms: int = 200,
        silence_duration_ms: int = 1000,
        max_segment_duration_seconds: int = 8,
        vogent_threshold: float = 0.5,
        model_dir: str = "/tmp/vogent_models",
    ):
        """
        Initialize Vogent Turn Detection.

        Args:
            whisper_model_size: Faster-whisper model size (tiny, base, small, medium, large)
            vad_reset_interval_seconds: Reset VAD internal state every N seconds to prevent drift
            speech_probability_threshold: Minimum probability to consider audio as speech (0.0-1.0)
            pre_speech_buffer_ms: Duration in ms to buffer before speech detection trigger
            silence_duration_ms: Duration of trailing silence in ms before ending a turn
            max_segment_duration_seconds: Maximum duration in seconds for a single audio segment
            vogent_threshold: Threshold for vogent turn completion probability (0.0-1.0)
            model_dir: Directory to store model files
        """
        super().__init__()

        # Configuration parameters
        self.whisper_model_size = whisper_model_size
        self.vad_reset_interval_seconds = vad_reset_interval_seconds
        self.speech_probability_threshold = speech_probability_threshold
        self.pre_speech_buffer_ms = pre_speech_buffer_ms
        self.silence_duration_ms = silence_duration_ms
        self.max_segment_duration_seconds = max_segment_duration_seconds
        self.vogent_threshold = vogent_threshold
        self.model_dir = model_dir

        # Audio buffering for processing
        self._audio_buffer = PcmData(
            sample_rate=RATE, channels=1, format=AudioFormat.F32
        )
        self._silence = Silence()
        self._pre_speech_buffer = PcmData(
            sample_rate=RATE, channels=1, format=AudioFormat.F32
        )
        self._active_segment: Optional[PcmData] = None
        self._trailing_silence_ms = self.silence_duration_ms

        # Producer-consumer pattern: audio packets go into buffer, background task processes them
        self._audio_queue: asyncio.Queue[Any] = asyncio.Queue()
        self._processing_task: Optional[asyncio.Task[Any]] = None
        self._shutdown_event = asyncio.Event()

        # Whisper model parameters
        self._whisper_model_size = whisper_model_size
        self._whisper_device = "cpu"
        self._whisper_language = "en"
        self._whisper_compute_type = "int8"

        # Will be set after warmup()
        self._whisper: WhisperModel | None = None
        self._vogent: VogentDetector | None = None
        self._vad_session: Optional[SileroVADSession] = None

    async def on_warmup(
        self,
    ) -> tuple[SileroVADSessionPool, WhisperModel, VogentDetector]:
        """Initialize models and prepare for turn detection."""
        # Ensure that model directory exists
        await asyncio.to_thread(os.makedirs, self.model_dir, exist_ok=True)

        # Load Silero VAD model
        vad_pool = await SileroVADSessionPool.load(self.model_dir)

        logger.info(f"Loading faster-whisper model: {self._whisper_model_size}")
        whisper_model = await asyncio.to_thread(
            lambda: WhisperModel(
                self._whisper_model_size,
                device=self._whisper_device,
                compute_type=self._whisper_compute_type,
            )
        )
        logger.info("Faster-whisper model loaded")

        logger.info("Loading Vogent turn detection model")
        # Note: compile_model=False to avoid torch.compile issues with edge cases
        vogent = await asyncio.to_thread(
            lambda: VogentDetector(
                compile_model=True,
                warmup=True,
                device=None,
                model_name="vogent/Vogent-Turn-80M",
                revision="main",
            )
        )
        logger.info("Vogent turn detection model loaded")

        return vad_pool, whisper_model, vogent

    def on_warmed_up(
        self, resources: tuple[SileroVADSessionPool, WhisperModel, VogentDetector]
    ) -> None:
        vad_pool, self._whisper, self._vogent = resources
        self._vad_session = vad_pool.session(self.vad_reset_interval_seconds)

    async def start(self):
        """
        Start the Vogent turn detection process after joining the call.
        """
        await super().start()

        # Start background processing task
        self._processing_task = asyncio.create_task(self._process_audio_loop())

    async def stop(self):
        """Stop turn detection and cleanup background task."""
        await super().stop()

        if self._processing_task:
            self._shutdown_event.set()
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
            self._processing_task = None

    async def process_audio(
        self,
        audio_data: PcmData,
        participant: Participant,
        conversation: Optional[Conversation],
    ) -> None:
        """
        Fast, non-blocking audio packet enqueueing.
        Actual processing happens in background task.
        """
        # Just enqueue the audio packet - fast and non-blocking
        await self._audio_queue.put((audio_data, participant, conversation))

    async def _process_audio_loop(self):
        """
        Background task that continuously processes audio from the queue.
        This is where the actual VAD and turn detection logic runs.
        """
        while not self._shutdown_event.is_set():
            try:
                # Wait for audio packet with timeout to allow shutdown
                audio_data, participant, conversation = await asyncio.wait_for(
                    self._audio_queue.get(), timeout=1.0
                )

                # Process the audio packet
                await self._process_audio_packet(audio_data, participant, conversation)

            except asyncio.TimeoutError:
                # Timeout is expected - continue loop to check shutdown
                continue
            except Exception as e:
                logger.error(f"Error processing audio: {e}")

    async def _process_audio_packet(
        self,
        audio_data: PcmData,
        participant: Participant,
        conversation: Optional[Conversation],
    ) -> None:
        """
        Process audio packet through VAD -> Whisper -> Vogent pipeline.

        This method:
        1. Buffers audio and processes in 512-sample chunks
        2. Uses VAD to detect speech
        3. Creates segments while people are speaking
        4. When reaching silence or max duration:
           - Transcribes segment with Whisper
           - Checks turn completion with Vogent (audio + text)

        Args:
            audio_data: PcmData object containing audio samples
            participant: Participant that's speaking
            conversation: Conversation history for context
        """
        if self._vad_session is None:
            raise ValueError("The VAD model is not initialized, call warmup() first")

        # Ensure audio is in the right format: 16kHz, float32
        audio_data = audio_data.resample(RATE).to_float32()
        self._audio_buffer = self._audio_buffer.append(audio_data)

        if len(self._audio_buffer.samples) < CHUNK:
            # Too small to process
            return

        # Split into 512-sample chunks
        audio_chunks = list(self._audio_buffer.chunks(CHUNK))
        self._audio_buffer = PcmData(
            sample_rate=RATE, channels=1, format=AudioFormat.F32
        )
        self._audio_buffer.append(audio_chunks[-1])  # Add back the last one
        # This ensures we handle the situation when audio data can't be divided by 512

        # Detect speech in small 512 chunks, gather to larger audio segments with speech
        for chunk in audio_chunks[:-1]:
            # Predict if this segment has speech
            # Run VAD in thread pool to avoid blocking event loop
            speech_probability = await asyncio.to_thread(
                self._vad_session.predict_speech, chunk
            )
            is_speech = speech_probability > self.speech_probability_threshold

            if self._active_segment is not None:
                # Add to the segment
                self._active_segment.append(chunk)

                if is_speech:
                    self._silence.speaking_chunks += 1
                    if self._silence.speaking_chunks > 3:
                        self._silence.trailing_silence_chunks = 0
                        self._silence.speaking_chunks = 0
                else:
                    self._silence.trailing_silence_chunks += 1

                trailing_silence_ms = (
                    self._silence.trailing_silence_chunks
                    * CHUNK
                    / RATE
                    * 1000
                    * 5  # DTX correction
                )
                long_silence = trailing_silence_ms > self._trailing_silence_ms
                max_duration_reached = (
                    self._active_segment.duration_ms
                    >= self.max_segment_duration_seconds * 1000
                )

                if long_silence or max_duration_reached:
                    # Expand to 8 seconds with either silence or historical
                    merged = PcmData(
                        sample_rate=RATE, channels=1, format=AudioFormat.F32
                    )
                    merged.append(self._pre_speech_buffer)
                    merged.append(self._active_segment)
                    merged = merged.tail(8, True, "start")

                    # Transcribe the segment with Whisper
                    transcription = await self._transcribe_segment(merged)

                    # Get previous line from conversation for context
                    prev_line = self._get_previous_line(conversation)

                    # Check if turn is complete using Vogent (multimodal: audio + text)
                    is_complete, confidence = await self._predict_turn_completed(
                        merged,
                        prev_line=prev_line,
                        curr_line=transcription,
                    )

                    if is_complete:
                        self._emit_end_turn_event(
                            participant=participant,
                            confidence=confidence,
                            trailing_silence_ms=trailing_silence_ms,
                            duration_ms=self._active_segment.duration_ms,
                        )
                        self._active_segment = None
                        self._silence = Silence()
                        # Add the merged segment to the speech buffer for next iteration
                        self._pre_speech_buffer = PcmData(
                            sample_rate=RATE, channels=1, format=AudioFormat.F32
                        )
                        self._pre_speech_buffer.append(merged)
                        self._pre_speech_buffer = self._pre_speech_buffer.tail(8)

            elif is_speech and self._active_segment is None:
                self._emit_start_turn_event(TurnStartedEvent(participant=participant))
                # Create a new segment
                self._active_segment = PcmData(
                    sample_rate=RATE, channels=1, format=AudioFormat.F32
                )
                self._active_segment.append(chunk)
                self._silence = Silence()
            else:
                # Keep last n audio packets in speech buffer
                self._pre_speech_buffer.append(chunk)
                self._pre_speech_buffer = self._pre_speech_buffer.tail(8)

    async def wait_for_processing_complete(self, timeout: float = 5.0) -> None:
        """Wait for all queued audio to be processed. Useful for testing."""
        start_time = time.time()
        while self._audio_queue.qsize() > 0 and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.01)

        # Give a small buffer for the processing to complete
        await asyncio.sleep(0.1)

    async def _transcribe_segment(self, pcm: PcmData) -> str:
        """
        Transcribe audio segment using faster-whisper.

        Args:
            pcm: PcmData containing audio samples

        Returns:
            Transcribed text
        """

        def _transcribe():
            if self._whisper is None:
                raise ValueError(
                    "Whisper model is not initialized, call warmup() first"
                )

            # All CPU-intensive work runs in thread pool
            audio_array = pcm.resample(16000).to_float32().samples
            segments, _ = self._whisper.transcribe(
                audio_array,
                language=self._whisper_language,
                beam_size=1,
                vad_filter=False,  # We already did VAD
            )
            # Collect all text segments
            text_parts = [segment.text.strip() for segment in segments]
            return " ".join(text_parts).strip()

        return await asyncio.to_thread(_transcribe)

    async def _predict_turn_completed(
        self,
        pcm: PcmData,
        prev_line: str,
        curr_line: str,
    ) -> tuple[bool, float]:
        """
        Predict whether the current turn is complete using Vogent.

        Args:
            pcm: PcmData containing audio samples
            prev_line: Previous speaker's text (for context)
            curr_line: Current speaker's text

        Returns:
            Tuple of (is_complete, confidence) where confidence is the probability
        """

        def _predict():
            if self._vogent is None:
                raise ValueError("Vogent is not initialized, call warmup() first")

            # All CPU-intensive work runs in thread pool
            audio_array = pcm.resample(16000).to_float32().tail(8, False).samples
            return self._vogent.predict(
                audio_array,
                prev_line=prev_line,
                curr_line=curr_line,
                sample_rate=16000,
                return_probs=True,
            )

        result = await asyncio.to_thread(_predict)
        confidence = float(result["prob_endpoint"])

        # Check if probability exceeds threshold
        is_complete = confidence > self.vogent_threshold
        logger.debug(
            f"Vogent probability: {confidence:.3f}, "
            f"threshold: {self.vogent_threshold}, is_complete: {is_complete}"
        )

        return is_complete, confidence

    def _get_previous_line(self, conversation: Optional[Conversation]) -> str:
        """
        Extract the previous speaker's line from conversation history.

        Args:
            conversation: Conversation object with message history

        Returns:
            Previous line text, or empty string if not available
        """
        if conversation is None or not conversation.messages:
            return ""

        # Get the last message that's not from the current speaker
        # Typically this would be the assistant or another user
        for message in reversed(conversation.messages):
            if message.content and message.content.strip():
                # Remove terminal punctuation for better vogent performance
                text = message.content.strip().rstrip(".!?")
                return text

        return ""
