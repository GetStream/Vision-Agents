"""
Fal Wizper STT Plugin for Stream

Provides real-time audio transcription and translation using fal-ai/wizper (Whisper v3).
This plugin integrates with Stream's audio processing pipeline to provide high-quality
speech-to-text capabilities.

Example usage:
    from vision_agents.plugins import fal

    # For transcription
    stt = fal.STT(task="transcribe")

    # For translation to Portuguese
    stt = fal.STT(task="translate", target_language="pt")

    @stt.on("transcript")
    async def on_transcript(text: str, user: Any, metadata: dict):
        print(f"Transcript: {text}")

    @stt.on("error")
    async def on_error(error: str):
        print(f"Error: {error}")
"""

import io
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional
import wave

import fal_client
from getstream.video.rtc.track_util import PcmData

from vision_agents.core import stt
from vision_agents.core.stt import TranscriptResponse

if TYPE_CHECKING:
    from vision_agents.core.edge.types import Participant

logger = logging.getLogger(__name__)


class STT(stt.STT):
    """
    Audio transcription and translation using fal-ai/wizper (Whisper v3).

    This plugin provides real-time speech-to-text capabilities using the fal-ai/wizper
    service, which is based on OpenAI's Whisper v3 model. It supports both transcription
    and translation tasks.

    Attributes:
        task: The task type - either "transcribe" or "translate"
        target_language: Target language code for translation (e.g., "pt" for Portuguese)
    """

    def __init__(
        self,
        task: str = "transcribe",
        target_language: Optional[str] = None,
        client: Optional[fal_client.AsyncClient] = None,
    ):
        """
        Initialize Wizper STT.

        Args:
            task: "transcribe" or "translate"
            target_language: Target language code (e.g., "pt" for Portuguese)
            client: Optional fal_client.AsyncClient instance for testing
        """
        super().__init__(provider_name="wizper")
        self.task = task
        self.sample_rate = 48000
        self.target_language = target_language
        self._fal_client = client if client is not None else fal_client.AsyncClient()

    def _pcm_to_wav_bytes(self, pcm_data: PcmData) -> bytes:
        """
        Convert PCM data to WAV format bytes.

        Args:
            pcm_data: PCM audio data from Stream's audio pipeline

        Returns:
            WAV format audio data as bytes
        """
        wav_buffer = io.BytesIO()

        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(pcm_data.samples.tobytes())

        wav_buffer.seek(0)
        return wav_buffer.read()

    async def process_audio(
        self,
        pcm_data: PcmData,
        participant: Optional["Participant"] = None,
    ):
        """
        Process audio through fal-ai/wizper for transcription.

        Args:
            pcm_data: The PCM audio data to process
            participant: Optional participant metadata
        """
        if self.closed:
            logger.warning("Wizper STT is closed, ignoring audio")
            return

        if pcm_data.samples.size == 0:
            logger.debug("No audio data to process")
            return

        try:
            logger.debug(
                "Sending speech audio to fal-ai/wizper",
                extra={"audio_bytes": pcm_data.samples.nbytes},
            )
            # Convert PCM to WAV format for upload
            wav_data = self._pcm_to_wav_bytes(pcm_data)

            # Create temporary file for upload
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(wav_data)
                temp_file.flush()
                temp_file_path = temp_file.name

            try:
                input_params = {
                    "task": "transcribe",  # TODO: make this dynamic, currently there's a bug in the fal-ai/wizper service where it only works with "transcribe"
                    "chunk_level": "segment",
                    "version": "3",
                }
                # Add language for translation
                if self.target_language is not None:
                    input_params["language"] = self.target_language

                # Upload file and get URL
                audio_url = await self._fal_client.upload_file(Path(temp_file_path))
                input_params["audio_url"] = audio_url

                # Use regular subscribe since streaming isn't supported
                result = await self._fal_client.subscribe(
                    "fal-ai/wizper", arguments=input_params
                )
                if "text" in result:
                    text = result["text"].strip()
                    if text:
                        response_metadata = TranscriptResponse()
                        self._emit_transcript_event(
                            text, participant, response_metadata
                        )
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except OSError:
                    pass

        except Exception as e:
            logger.error(f"Wizper processing error: {str(e)}")
            self._emit_error_event(e, "Wizper processing")

    async def close(self):
        """Close the Wizper STT service and release any resources."""
        if self.closed:
            logger.debug("Wizper STT service already closed")
            return

        logger.info("Closing Wizper STT service")
        await super().close()
