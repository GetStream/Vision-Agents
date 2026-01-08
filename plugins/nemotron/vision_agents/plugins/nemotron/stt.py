import base64
import logging
import time
from typing import Optional

import httpx
from getstream.video.rtc.track_util import AudioFormat, PcmData

from vision_agents.core import stt
from vision_agents.core.edge.types import Participant
from vision_agents.core.stt.events import TranscriptResponse

logger = logging.getLogger(__name__)

RATE = 16000
MIN_BUFFER_DURATION_MS = 500
MAX_BUFFER_DURATION_MS = 8000
PROCESS_INTERVAL_MS = 1000


class STT(stt.STT):
    """
    NVIDIA Nemotron Speech-to-Text client.

    Connects to a Nemotron ASR server for transcription.
    See plugins/nemotron/server for the server component.

    Audio is buffered and sent to the server periodically.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8765",
        timeout: float = 30.0,
    ):
        """
        Initialize Nemotron STT client.

        Args:
            server_url: URL of the Nemotron ASR server
            timeout: HTTP request timeout in seconds
        """
        super().__init__(provider_name="nemotron")

        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._audio_buffer = PcmData(
            sample_rate=RATE, channels=1, format=AudioFormat.F32
        )
        self._last_process_time = time.time()

    async def start(self):
        """Start the STT client and verify server connection."""
        await super().start()
        self._client = httpx.AsyncClient(timeout=self.timeout)

        response = await self._client.get(f"{self.server_url}/health")
        response.raise_for_status()
        health = response.json()
        if not health.get("model_loaded"):
            raise RuntimeError("Nemotron server model not loaded")
        logger.info("Connected to Nemotron server")

    async def process_audio(
        self,
        pcm_data: PcmData,
        participant: Optional[Participant] = None,
    ):
        """
        Process audio data through Nemotron server for transcription.

        Args:
            pcm_data: The PCM audio data to process
            participant: Optional participant metadata
        """
        if self.closed:
            logger.warning("Nemotron STT is closed, ignoring audio")
            return

        if self._client is None:
            raise ValueError("STT not started, call start() first")

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
        """Send buffered audio to server for transcription."""
        buffer_to_process = self._audio_buffer

        self._audio_buffer = PcmData(
            sample_rate=RATE, channels=1, format=AudioFormat.F32
        )
        self._last_process_time = time.time()

        pcm = buffer_to_process.resample(RATE).to_float32()
        audio_samples = pcm.samples

        if audio_samples.size == 0:
            return

        audio_bytes = audio_samples.tobytes()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        if self._client is None:
            raise ValueError("STT client not initialized")

        start_time = time.time()

        response = await self._client.post(
            f"{self.server_url}/transcribe",
            json={"audio_base64": audio_base64, "sample_rate": RATE},
        )
        response.raise_for_status()
        result = response.json()

        text = result.get("text", "").strip()
        server_time_ms = result.get("processing_time_ms", 0)
        total_time_ms = (time.time() - start_time) * 1000

        if participant is None:
            participant = Participant(original=None, user_id="unknown")

        if text:
            response_meta = TranscriptResponse(
                language="en",
                processing_time_ms=total_time_ms,
                audio_duration_ms=buffer_to_process.duration_ms,
                model_name="nemotron-speech",
                other={"server_processing_time_ms": server_time_ms},
            )
            self._emit_transcript_event(text, participant, response_meta)

    async def close(self):
        """Close the STT client."""
        await super().close()
        if self._client:
            await self._client.aclose()
            self._client = None
