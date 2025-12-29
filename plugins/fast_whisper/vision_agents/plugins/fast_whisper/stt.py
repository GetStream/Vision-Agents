import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Optional

from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment, TranscriptionInfo
from getstream.video.rtc.track_util import AudioFormat, PcmData
from numpy.typing import NDArray
from vision_agents.core import stt
from vision_agents.core.edge.types import Participant
from vision_agents.core.stt.events import TranscriptResponse
from vision_agents.core.warmup import Warmable

logger = logging.getLogger(__name__)

# Audio processing constants
RATE = 16000  # Sample rate in Hz (16kHz)
MIN_BUFFER_DURATION_MS = 1000  # Minimum buffer duration before processing (1 second)
MAX_BUFFER_DURATION_MS = (
    8000  # Maximum buffer duration before forcing processing (8 seconds)
)
PROCESS_INTERVAL_MS = 2000  # Process buffer every 2 seconds if it has content


class STT(stt.STT, Warmable[Optional[WhisperModel]]):
    """
    Faster-Whisper Speech-to-Text implementation.

    This implementation uses faster-whisper for offline transcription.
    Audio is buffered and processed periodically to provide near-real-time
    transcription results.

    Since faster-whisper is not a streaming STT, this implementation:
    1. Buffers incoming audio chunks
    2. Processes the buffer periodically (every 2 seconds) or when it reaches max duration
    3. Emits partial transcripts for individual segments
    4. Emits final transcripts for complete buffers
    """

    def __init__(
        self,
        model_size: Literal["tiny", "base", "small", "medium", "large"] = "tiny",
        language: Optional[str] = "en",
        device: Literal["cpu", "cuda"] = "cpu",
        client: Optional[WhisperModel] = None,
    ):
        """
        Initialize Faster-Whisper STT.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            language: Language code (e.g., "en", "es", "fr") or None for auto-detect
            device: Device to run on ("cpu" or "cuda")
            client: Optional pre-initialized WhisperModel instance
        """
        super().__init__(provider_name="faster_whisper")

        self.model_size = model_size
        self.language = language
        self.device = device
        self.compute_type = "int8"

        self.whisper = client

        self._audio_buffer = PcmData(
            sample_rate=RATE, channels=1, format=AudioFormat.F32
        )
        self._last_process_time = time.time()
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def on_warmup(self) -> Optional[WhisperModel]:
        if self.whisper is None:
            logger.info(f"Loading faster-whisper model: {self.model_size}")
            # Load whisper in thread pool to avoid blocking event loop
            loop = asyncio.get_running_loop()
            whisper = await loop.run_in_executor(
                self._executor,
                lambda: WhisperModel(
                    self.model_size, device=self.device, compute_type=self.compute_type
                ),
            )
            logger.info("Faster-whisper model loaded")
            return whisper

        # The WhisperModel is already provided on init, no need to load it
        return None

    def on_warmed_up(self, whisper: Optional[WhisperModel]) -> None:
        if self.whisper is None:
            self.whisper = whisper

    async def process_audio(
        self,
        pcm_data: PcmData,
        participant: Optional[Participant] = None,
    ):
        """
        Process audio data through faster-whisper for transcription.

        Audio is buffered and processed periodically to provide transcription results.

        Args:
            pcm_data: The PCM audio data to process
            participant: Optional participant metadata
        """
        if self.closed:
            logger.warning("Faster-Whisper STT is closed, ignoring audio")
            return

        if self.whisper is None:
            raise ValueError("Whisper model not loaded, call warmup() first")

        # Check for empty audio
        if pcm_data.samples.size == 0:
            return

        try:
            # Ensure audio is in the right format: 16kHz, float32

            audio_data = pcm_data.resample(RATE).to_float32()
            self._audio_buffer = self._audio_buffer.append(audio_data)

            current_time = time.time()
            buffer_duration_ms = self._audio_buffer.duration_ms
            buffer_size = self._audio_buffer.samples.size
            time_since_last_process = (current_time - self._last_process_time) * 1000

            should_process = (
                buffer_duration_ms >= MIN_BUFFER_DURATION_MS
                and buffer_size > 0
                and (
                    time_since_last_process >= PROCESS_INTERVAL_MS
                    or buffer_duration_ms >= MAX_BUFFER_DURATION_MS
                )
            )

            if should_process:
                await self._process_buffer(participant)

        except Exception as e:
            logger.exception("Error buffering audio for faster-whisper")
            self._emit_error_event(
                e, context="buffering_audio", participant=participant
            )

    async def _process_buffer(self, participant: Optional[Participant] = None):
        """
        Process the current audio buffer through faster-whisper.

        Args:
            participant: Optional participant metadata
        """

        # Extract buffer to process
        buffer_to_process = self._audio_buffer

        self._audio_buffer = PcmData(
            sample_rate=RATE, channels=1, format=AudioFormat.F32
        )
        self._last_process_time = time.time()

        # Ensure it's 16kHz and f32 format
        pcm = buffer_to_process.resample(RATE).to_float32()
        audio_array = pcm.samples

        if audio_array.size == 0:
            return

        start_time = time.time()

        try:
            segments, info = await self._transcribe(audio_array=audio_array)
        except Exception as e:
            logger.error(
                "Error processing audio buffer with faster-whisper", exc_info=e
            )
            self._emit_error_event(e, context="transcription", participant=participant)
            return

        processing_time_ms = (time.time() - start_time) * 1000

        # Create default participant if none provided
        if participant is None:
            participant = Participant(original=None, user_id="unknown")

        # Process segments
        text_parts = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                text_parts.append(text)

                # Emit partial transcript for each segment
                response = TranscriptResponse(
                    confidence=segment.no_speech_prob
                    if hasattr(segment, "no_speech_prob")
                    else None,
                    language=info.language
                    if hasattr(info, "language")
                    else self.language,
                    processing_time_ms=processing_time_ms,
                    audio_duration_ms=buffer_to_process.duration_ms,
                    model_name=f"faster-whisper-{self.model_size}",
                )
                self._emit_partial_transcript_event(text, participant, response)

        # Emit final transcript for the complete buffer
        if text_parts:
            full_text = " ".join(text_parts).strip()
            response = TranscriptResponse(
                confidence=None,  # faster-whisper doesn't provide overall confidence
                language=info.language if hasattr(info, "language") else self.language,
                processing_time_ms=processing_time_ms,
                audio_duration_ms=buffer_to_process.duration_ms,
                model_name=f"faster-whisper-{self.model_size}",
            )
            self._emit_transcript_event(full_text, participant, response)

    async def close(self):
        """Close the STT and cleanup resources."""
        await super().close()
        self._executor.shutdown(wait=False)

    async def _transcribe(
        self, audio_array: NDArray
    ) -> tuple[list[Segment], TranscriptionInfo]:
        if self.whisper is None:
            raise ValueError("Whisper model not loaded, call warmup() first")

        whisper = self.whisper  # Type narrowing for closure

        def _worker():
            segments, info = whisper.transcribe(
                audio_array,
                language=self.language,
                beam_size=1,
                vad_filter=False,  # Let faster-whisper handle VAD if needed
            )
            # Evaluate the generator in the background thread
            return list(segments), info

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, _worker)
