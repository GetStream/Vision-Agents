import asyncio
import logging
import os
import time
from typing import Optional

import numpy as np
from fish_audio_sdk import ASRRequest, Session
from getstream.video.rtc.track_util import PcmData
from vision_agents.core import stt
from vision_agents.core.edge.types import Participant
from vision_agents.core.stt import TranscriptResponse

logger = logging.getLogger(__name__)

MIN_DURATION_MS = 1000.0


class STT(stt.STT):
    """
    Fish Audio Speech-to-Text implementation.

    Fish Audio requires at least 1 second of audio per request. Audio is buffered
    per participant until the minimum duration is reached, then sent to the API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        language: Optional[str] = None,
        client: Optional[Session] = None,
    ):
        super().__init__(provider_name="fish")

        if not api_key:
            api_key = os.environ.get("FISH_API_KEY")

        if client is not None:
            self.client = client
        else:
            if not api_key:
                raise ValueError("api_key is required")
            self.client = Session(api_key)

        self.language = language
        self._buffers: dict[str, PcmData] = {}
        self._buffer_participant: dict[str, Participant] = {}

    async def process_audio(
        self,
        pcm_data: PcmData,
        participant: Participant,
    ):
        """
        Process audio data through Fish Audio for transcription.

        Fish Audio operates in synchronous mode - it processes audio immediately and
        returns results to the base class for event emission.

        Args:
            pcm_data: The PCM audio data to process.
            user_metadata: Additional metadata about the user or session.

        Returns:
            List of tuples (is_final, text, metadata) representing transcription results,
            or None if no results are available. Fish Audio returns final results only.
        """
        if self.closed:
            logger.warning("Fish Audio STT is closed, ignoring audio")
            return None

        if pcm_data.samples is None:
            logger.warning("No audio samples to process")
            return None

        if isinstance(pcm_data.samples, np.ndarray) and pcm_data.samples.size == 0:
            logger.debug("Received empty audio data")
            return None

        key = participant.user_id
        if key not in self._buffers:
            self._buffers[key] = pcm_data.copy()
            self._buffer_participant[key] = participant
        else:
            self._buffers[key].append(pcm_data)

        buf = self._buffers[key]
        if buf.duration_ms < MIN_DURATION_MS:
            return None

        await self._send_buffer(key)

    async def _send_buffer(self, key: str) -> None:
        buf = self._buffers.get(key)
        participant = self._buffer_participant.get(key)
        if buf is None or participant is None or buf.duration_ms < MIN_DURATION_MS:
            return

        pcm_to_send = buf.copy()
        buf.clear()

        try:
            start_time = time.perf_counter()
            wav_data = pcm_to_send.to_wav_bytes()
            asr_request = ASRRequest(
                audio=wav_data,
                language=self.language,
                ignore_timestamps=True,
            )
            logger.debug(
                "Sending audio to Fish Audio ASR",
                extra={"audio_bytes": len(wav_data)},
            )
            response = await asyncio.to_thread(self.client.asr, asr_request)
            transcript_text = response.text.strip()

            if not transcript_text:
                logger.debug(
                    "No transcript from Fish Audio (duration_ms=%.0f)",
                    pcm_to_send.duration_ms,
                )
                return

            processing_time_ms = (time.perf_counter() - start_time) * 1000
            response_metadata = TranscriptResponse(
                audio_duration_ms=response.duration,
                language=self.language or "auto",
                model_name="fish-audio-asr",
                processing_time_ms=processing_time_ms,
            )
            logger.debug(
                "Received transcript from Fish Audio",
                extra={
                    "text_length": len(transcript_text),
                    "duration_ms": response.duration,
                },
            )
            self._emit_transcript_event(transcript_text, participant, response_metadata)

        except Exception:
            logger.exception("Error during Fish Audio transcription")
            raise

    async def clear(self) -> None:
        for key in list(self._buffers):
            if self._buffers[key].duration_ms >= MIN_DURATION_MS:
                await self._send_buffer(key)
            else:
                self._buffers[key].clear()
        await super().clear()
