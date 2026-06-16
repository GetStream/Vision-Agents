import asyncio
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from getstream.video.rtc.track_util import AudioFormat, PcmData
from numpy.typing import NDArray
from vision_agents.core import stt
from vision_agents.core.edge.types import Participant
from vision_agents.core.stt import TranscriptResponse
from vision_agents.core.warmup import Warmable

logger = logging.getLogger(__name__)

# Audio processing constants
RATE = 16000  # Sample rate in Hz (16kHz) expected by SenseVoice
MIN_BUFFER_DURATION_MS = 1000  # Minimum buffer duration before processing (1 second)
MAX_BUFFER_DURATION_MS = (
    8000  # Maximum buffer duration before forcing processing (8 seconds)
)
PROCESS_INTERVAL_MS = 2000  # Process buffer every 2 seconds if it has content

# SenseVoice prepends meta tags like ``<|en|><|HAPPY|><|Speech|><|withitn|>`` to
# the transcription. These sets classify the language and emotion tags.
_LANGUAGE_TAGS = {"zh", "en", "yue", "ja", "ko", "nospeech"}
_EMOTION_TAGS = {
    "HAPPY",
    "SAD",
    "ANGRY",
    "NEUTRAL",
    "FEARFUL",
    "DISGUSTED",
    "SURPRISED",
    "EMO_UNKNOWN",
}


def _parse_sensevoice_tags(raw_text: str) -> tuple[Optional[str], Optional[str]]:
    """Extract the detected language and emotion from SenseVoice meta tags."""
    language: Optional[str] = None
    emotion: Optional[str] = None
    for tag in re.findall(r"<\|([^|>]+)\|>", raw_text):
        if tag in _LANGUAGE_TAGS and language is None:
            language = tag
        elif tag in _EMOTION_TAGS and emotion is None:
            emotion = tag
    return language, emotion


class STT(stt.STT, Warmable[Optional[AutoModel]]):
    """
    FunASR Speech-to-Text implementation.

    Self-hosted transcription using FunASR's SenseVoice model. SenseVoice is
    non-autoregressive (single forward pass), supports 50+ languages with
    auto-detection, and emits emotion tags that are surfaced on the transcript.

    Since SenseVoice is not a streaming STT, this implementation:
    1. Buffers incoming audio chunks
    2. Processes the buffer periodically (every 2 seconds) or when it reaches max duration
    3. Emits a final transcript for each processed buffer
    """

    def __init__(
        self,
        model: str = "iic/SenseVoiceSmall",
        language: str = "auto",
        device: str = "cpu",
        vad_model: Optional[str] = "fsmn-vad",
        client: Optional[AutoModel] = None,
    ):
        """
        Initialize FunASR STT.

        Args:
            model: FunASR model id (e.g. "iic/SenseVoiceSmall")
            language: Language code ("zh", "en", "yue", "ja", "ko") or "auto"
            device: Device to run on ("cpu" or "cuda")
            vad_model: VAD model used to segment long audio, or None to disable
            client: Optional pre-initialized AutoModel instance
        """
        super().__init__(provider_name="funasr")

        self.model_name = model
        self.language = language
        self.device = device
        self.vad_model = vad_model

        self.model = client

        self._audio_buffer = PcmData(
            sample_rate=RATE, channels=1, format=AudioFormat.F32
        )
        self._last_process_time = time.time()
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def on_warmup(self) -> Optional[AutoModel]:
        if self.model is None:
            logger.info(f"Loading FunASR model: {self.model_name}")
            # Load the model in a thread pool to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            model = await loop.run_in_executor(
                self._executor,
                lambda: AutoModel(
                    model=self.model_name,
                    vad_model=self.vad_model,
                    device=self.device,
                    disable_update=True,
                ),
            )
            logger.info("FunASR model loaded")
            return model

        # The AutoModel is already provided on init, no need to load it
        return None

    def on_warmed_up(self, model: Optional[AutoModel]) -> None:
        if self.model is None:
            self.model = model

    async def process_audio(
        self,
        pcm_data: PcmData,
        participant: Participant,
    ):
        """
        Process audio data through FunASR for transcription.

        Audio is buffered and processed periodically to provide transcription results.

        Args:
            pcm_data: The PCM audio data to process
            participant: Participant metadata
        """
        if self.closed:
            logger.warning("FunASR STT is closed, ignoring audio")
            return

        if self.model is None:
            raise ValueError("FunASR model not loaded, call warmup() first")

        if pcm_data.samples.size == 0:
            return

        try:
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

        except Exception:
            logger.exception("Error buffering audio for FunASR")

    async def _process_buffer(self, participant: Participant):
        """Process the current audio buffer through FunASR."""
        buffer_to_process = self._audio_buffer

        self._audio_buffer = PcmData(
            sample_rate=RATE, channels=1, format=AudioFormat.F32
        )
        self._last_process_time = time.time()

        pcm = buffer_to_process.resample(RATE).to_float32()
        audio_array = pcm.samples

        if audio_array.size == 0:
            return

        start_time = time.time()

        try:
            raw_text = await self._transcribe(audio_array=audio_array)
        except Exception:
            logger.exception("Error processing audio buffer with FunASR")
            return

        processing_time_ms = (time.time() - start_time) * 1000

        text = rich_transcription_postprocess(raw_text).strip()
        if not text:
            return

        language, emotion = _parse_sensevoice_tags(raw_text)
        response = TranscriptResponse(
            language=language or (self.language if self.language != "auto" else None),
            processing_time_ms=processing_time_ms,
            audio_duration_ms=buffer_to_process.duration_ms,
            model_name=self.model_name,
            other={"emotion": emotion} if emotion else None,
        )
        self._emit_transcript_event(text, participant, response, mode="final")

    async def close(self):
        """Close the STT and cleanup resources."""
        await super().close()
        self._executor.shutdown(wait=False)

    async def _transcribe(self, audio_array: NDArray) -> str:
        if self.model is None:
            raise ValueError("FunASR model not loaded, call warmup() first")

        model = self.model  # Type narrowing for closure

        def _worker() -> str:
            res = model.generate(
                input=audio_array,
                cache={},
                language=self.language,
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )
            if not res:
                return ""
            return res[0]["text"]

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, _worker)
