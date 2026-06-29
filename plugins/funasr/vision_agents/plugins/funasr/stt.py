import asyncio
import logging
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
RATE = 16000  # Sample rate in Hz (16kHz)
MIN_BUFFER_DURATION_MS = 1000  # Minimum buffer duration before processing (1 second)
MAX_BUFFER_DURATION_MS = 8000  # Maximum buffer duration before forcing processing
PROCESS_INTERVAL_MS = 2000  # Process buffer every 2 seconds if it has content


class STT(stt.STT, Warmable[Optional[AutoModel]]):
    """
    FunASR Speech-to-Text implementation (local, self-hosted).

    Uses `FunASR <https://github.com/modelscope/FunASR>`_ (SenseVoice / Fun-ASR-Nano /
    Paraformer) for offline transcription — strong on Chinese, Cantonese, Japanese,
    Korean and more. The model runs locally on CPU or CUDA; no API key is required.

    FunASR is not a streaming STT, so this implementation:

    1. Buffers incoming audio chunks.
    2. Processes the buffer periodically (every 2 seconds) or when it reaches max
       duration.
    3. Emits a final transcript for each processed buffer.
    """

    def __init__(
        self,
        model: str = "iic/SenseVoiceSmall",
        language: str = "auto",
        device: str = "cpu",
        use_itn: bool = True,
        client: Optional[AutoModel] = None,
    ):
        """
        Initialize FunASR STT.

        Args:
            model: FunASR model id. Defaults to ``iic/SenseVoiceSmall`` (fast,
                multilingual, CPU-friendly). Use ``FunAudioLLM/Fun-ASR-Nano-2512``
                (the flagship LLM-ASR model) on GPU, or ``paraformer-zh`` for Chinese.
            language: Language hint (e.g. ``"zh"``, ``"en"``) or ``"auto"`` to detect.
            device: Device to run on (``"cpu"`` or ``"cuda"``).
            use_itn: Apply inverse text normalization.
            client: Optional pre-initialized FunASR ``AutoModel`` instance.
        """
        super().__init__(provider_name="funasr")

        self.model_id = model
        self.language = language
        self.device = device
        self.use_itn = use_itn

        self.funasr = client

        self._audio_buffer = PcmData(
            sample_rate=RATE, channels=1, format=AudioFormat.F32
        )
        self._last_process_time = time.time()
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def on_warmup(self) -> Optional[AutoModel]:
        if self.funasr is None:
            logger.info(f"Loading FunASR model: {self.model_id}")
            loop = asyncio.get_running_loop()
            model = await loop.run_in_executor(
                self._executor,
                lambda: AutoModel(
                    model=self.model_id, device=self.device, disable_update=True
                ),
            )
            logger.info("FunASR model loaded")
            return model

        # The model is already provided on init, no need to load it
        return None

    def on_warmed_up(self, model: Optional[AutoModel]) -> None:
        if self.funasr is None:
            self.funasr = model

    async def process_audio(
        self,
        pcm_data: PcmData,
        participant: Participant,
    ):
        """
        Buffer audio and process it periodically through FunASR.

        Args:
            pcm_data: The PCM audio data to process.
            participant: Optional participant metadata.
        """
        if self.closed:
            logger.warning("FunASR STT is closed, ignoring audio")
            return

        if self.funasr is None:
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

        except ValueError:
            logger.exception("Invalid PCM audio while buffering for FunASR")
        except RuntimeError:
            logger.exception("Audio buffering/runtime error for FunASR")

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
            text = await self._transcribe(audio_array=audio_array)
        except (RuntimeError, ValueError, KeyError, IndexError, TypeError):
            logger.exception("Error processing audio buffer with FunASR")
            return

        processing_time_ms = (time.time() - start_time) * 1000

        if text:
            response = TranscriptResponse(
                confidence=None,
                language=self.language,
                processing_time_ms=processing_time_ms,
                audio_duration_ms=buffer_to_process.duration_ms,
                model_name=f"funasr-{self.model_id}",
            )
            self._emit_transcript_event(text, participant, response, mode="final")

    async def close(self):
        """Close the STT and clean up resources."""
        try:
            await super().close()
        finally:
            self._executor.shutdown(wait=False)

    async def _transcribe(self, audio_array: NDArray) -> str:
        if self.funasr is None:
            raise ValueError("FunASR model not loaded, call warmup() first")

        model = self.funasr  # Type narrowing for closure

        def _worker() -> str:
            res = model.generate(
                input=audio_array,
                language=self.language,
                use_itn=self.use_itn,
            )
            if not res:
                return ""
            # SenseVoice output carries tags like <|zh|><|NEUTRAL|>...; strip them.
            return rich_transcription_postprocess(res[0]["text"]).strip()

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, _worker)
