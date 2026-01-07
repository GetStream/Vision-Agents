import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, Optional

from getstream.video.rtc.track_util import AudioFormat, PcmData

from vision_agents.core import stt
from vision_agents.core.edge.types import Participant
from vision_agents.core.stt.events import TranscriptResponse
from vision_agents.core.warmup import Warmable

logger = logging.getLogger(__name__)

RATE = 16000
MIN_BUFFER_DURATION_MS = 500
MAX_BUFFER_DURATION_MS = 8000
PROCESS_INTERVAL_MS = 1000

CHUNK_SIZE_TO_CONTEXT: dict[str, tuple[int, int]] = {
    "80ms": (70, 0),
    "160ms": (70, 1),
    "560ms": (70, 6),
    "1120ms": (70, 13),
}


class STT(stt.STT, Warmable[Optional[Any]]):
    """
    NVIDIA Nemotron Speech-to-Text implementation.

    Uses NeMo's cache-aware FastConformer + RNN-T architecture for
    high-quality English transcription with punctuation and capitalization.

    Audio is buffered and processed periodically to provide transcription results.
    """

    def __init__(
        self,
        model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b",
        chunk_size: Literal["80ms", "160ms", "560ms", "1120ms"] = "560ms",
        device: Literal["cpu", "cuda"] = "cpu",
        client: Optional[Any] = None,
    ):
        """
        Initialize Nemotron STT.

        Args:
            model_name: HuggingFace model name for Nemotron Speech
            chunk_size: Processing chunk size affecting latency-accuracy tradeoff.
                        Smaller chunks = lower latency but slightly higher WER.
                        Options: 80ms, 160ms, 560ms, 1120ms
            device: Device to run on ("cpu" or "cuda")
            client: Optional pre-initialized ASRModel instance
        """
        super().__init__(provider_name="nemotron")

        self.model_name = model_name
        self.chunk_size = chunk_size
        self.device = device
        self._model = client
        self._att_context_size = list(CHUNK_SIZE_TO_CONTEXT[chunk_size])
        self._audio_buffer = PcmData(
            sample_rate=RATE, channels=1, format=AudioFormat.F32
        )
        self._last_process_time = time.time()
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def on_warmup(self) -> Optional[Any]:
        if self._model is not None:
            return None

        import nemo.collections.asr as nemo_asr

        logger.info(f"Loading Nemotron model: {self.model_name}")

        def _load_model():
            model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_name)
            if self.device == "cuda":
                model = model.cuda()
            return model

        loop = asyncio.get_running_loop()
        model = await loop.run_in_executor(self._executor, _load_model)
        logger.info("Nemotron model loaded")
        return model

    def on_warmed_up(self, model: Optional[Any]) -> None:
        if self._model is None:
            self._model = model

    async def process_audio(
        self,
        pcm_data: PcmData,
        participant: Optional[Participant] = None,
    ):
        """
        Process audio data through Nemotron for transcription.

        Audio is buffered and processed periodically to provide transcription results.

        Args:
            pcm_data: The PCM audio data to process
            participant: Optional participant metadata
        """
        if self.closed:
            logger.warning("Nemotron STT is closed, ignoring audio")
            return

        if self._model is None:
            raise ValueError("Model not loaded, call warmup() first")

        if pcm_data.samples.size == 0:
            return

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

    async def _process_buffer(self, participant: Optional[Participant] = None):
        """Process the current audio buffer through Nemotron."""
        buffer_to_process = self._audio_buffer

        self._audio_buffer = PcmData(
            sample_rate=RATE, channels=1, format=AudioFormat.F32
        )
        self._last_process_time = time.time()

        pcm = buffer_to_process.resample(RATE).to_float32()
        audio_samples = pcm.samples

        if audio_samples.size == 0:
            return

        start_time = time.time()

        transcripts = await self._transcribe(audio_samples)

        processing_time_ms = (time.time() - start_time) * 1000

        if participant is None:
            participant = Participant(original=None, user_id="unknown")

        if transcripts:
            full_text = " ".join(transcripts).strip()
            if full_text:
                response = TranscriptResponse(
                    language="en",
                    processing_time_ms=processing_time_ms,
                    audio_duration_ms=buffer_to_process.duration_ms,
                    model_name=self.model_name,
                )
                self._emit_transcript_event(full_text, participant, response)

    async def _transcribe(self, audio_samples) -> list[str]:
        if self._model is None:
            raise ValueError("Model not loaded, call warmup() first")

        model = self._model

        def _worker():
            transcriptions = model.transcribe([audio_samples])
            if isinstance(transcriptions, tuple):
                transcriptions = transcriptions[0]
            return transcriptions

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, _worker)

    async def close(self):
        """Close the STT and cleanup resources."""
        await super().close()
        self._executor.shutdown(wait=False)
