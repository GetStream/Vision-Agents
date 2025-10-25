
import asyncio
import base64
import io
import logging
import os
import tempfile
import time
import wave
from pathlib import Path
from typing import Dict, Optional, Any

import fal_client
import numpy as np
from getstream.audio.utils import resample_audio
from getstream.video.rtc.track_util import PcmData

from vision_agents.core.agents import Conversation
from vision_agents.core.edge.types import Participant
from vision_agents.core.utils.utils import to_mono
from vision_agents.core.turn_detection.turn_detection import (
    TurnDetector,
    TurnEvent,
    TurnEventData,
)


def _resample(samples: np.ndarray) -> np.ndarray:
    """Resample audio from 48 kHz to 16 kHz."""
    return resample_audio(samples, 48000, 16000).astype(np.int16)


class TurnDetection(TurnDetector):
    """
    Turn detection implementation using FAL AI + smart-turn model.
    https://github.com/pipecat-ai/smart-turn
    https://pypi.org/project/fal-client/
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        buffer_in_seconds: float = 2.0, # seconds
        confidence_threshold: float = 0.5,
        sample_rate: int = 16000,
        channels: int = 1,
    ):
        """
        Initialize Smart Turn detection.

        Args:
            api_key: FAL API key (if None, uses FAL_KEY env var)
            buffer_in_seconds: Duration in seconds to buffer audio before processing
            confidence_threshold: Probability threshold for "complete" predictions
            sample_rate: Audio sample rate (Hz)
            channels: Number of audio channels
        """

        super().__init__(
            confidence_threshold=confidence_threshold, provider_name="SmartTurnDetection"
        )
        self.logger = logging.getLogger("SmartTurnDetection")
        self.api_key = api_key
        self.buffer_duration = buffer_in_seconds
        self.sample_rate = sample_rate
        self.channels = channels

        # Audio buffering per user - stores resampled samples (16kHz int16)
        self._user_buffers: Dict[str, list[np.ndarray]] = {}
        self._user_last_audio: Dict[str, float] = {}
        self._current_speaker: Optional[str] = None

        # Processing state - queue for ordered processing
        self._processing_queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._temp_dir = Path(tempfile.gettempdir()) / "smart_turn_detection"
        self._temp_dir.mkdir(exist_ok=True)

        # Configure FAL client
        if self.api_key:
            os.environ["FAL_KEY"] = self.api_key

        self.logger.info(
            f"Initialized Smart Turn detection (buffer: {buffer_in_seconds}s, threshold: {confidence_threshold})"
        )

    def start(self) -> None:
        """Start turn detection and processing worker."""
        super().start()
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._processing_worker())
            self.logger.info("Started processing worker task")

    async def _processing_worker(self) -> None:
        """Worker task that processes audio chunks from the queue in order."""
        self.logger.info("Processing worker started")
        while self.is_active:
            try:
                # Wait for items with timeout so we can check is_active periodically
                user_id, samples = await asyncio.wait_for(
                    self._processing_queue.get(), timeout=1.0
                )
                await self._process_extracted_audio(user_id, samples)
                self._processing_queue.task_done()
            except asyncio.TimeoutError:
                # No items in queue, continue loop to check is_active
                continue
            except Exception as e:
                self.logger.error("Error in processing worker: %s", e, exc_info=True)
        self.logger.info("Processing worker stopped")

    async def process_audio(
        self,
        audio_data: PcmData,
        participant: Participant,
        conversation: Optional[Conversation] = None,
    ) -> None:
        if not self.is_active:
            return

        user_id = participant.user_id
        # Validate sample format
        valid_formats = ["int16", "s16", "pcm_s16le"]
        if audio_data.format not in valid_formats:
            self.logger.error(
                f"Invalid sample format: {audio_data.format}. Expected one of {valid_formats}."
            )
            return
        if (
            not isinstance(audio_data.samples, np.ndarray)
            or audio_data.samples.dtype != np.int16
        ):
            self.logger.error(
                f"Invalid sample dtype: {audio_data.samples.dtype}. Expected int16."
            )
            return

        # Resample from 48 kHz to 16 kHz
        samples = _resample(audio_data.samples)

        # Initialize buffer for new user
        self._user_buffers.setdefault(user_id, [])
        self._user_last_audio[user_id] = time.time()

        # Append samples to buffer
        self._user_buffers[user_id].append(samples)

        # Calculate total buffered samples
        total_samples = sum(len(chunk) for chunk in self._user_buffers[user_id])
        required_samples = int(self.buffer_duration * self.sample_rate)

        if total_samples >= required_samples:
            # Extract the data from buffer immediately, before spawning the task
            # This allows the buffer to accumulate more data while the task processes
            audio_buffer = self._user_buffers[user_id]
            
            # Collect samples until we have enough
            process_chunks = []
            samples_collected = 0
            while audio_buffer and samples_collected < required_samples:
                chunk = audio_buffer.pop(0)
                samples_needed = required_samples - samples_collected
                
                if len(chunk) <= samples_needed:
                    # Use the entire chunk
                    process_chunks.append(chunk)
                    samples_collected += len(chunk)
                else:
                    # Split the chunk - take what we need and put the rest back
                    process_chunks.append(chunk[:samples_needed])
                    audio_buffer.insert(0, chunk[samples_needed:])
                    samples_collected += samples_needed
            
            # Concatenate all chunks into a single array
            process_samples = np.concatenate(process_chunks)
            
            # Put in queue for ordered processing by worker task
            await self._processing_queue.put((user_id, process_samples))

    async def _process_extracted_audio(self, user_id: str, process_samples: np.ndarray) -> None:
        """
        Process extracted audio samples for a specific user through FAL API.

        Args:
            user_id: ID of the user whose audio to process
            process_samples: The audio samples (np.ndarray of int16) to process
        """
        try:
            # Create WAV in memory
            wav_bytes = self._create_wav_bytes(process_samples)
            
            # Save to temporary file for upload
            temp_file = self._temp_dir / f"audio_{user_id}_{int(time.time() * 1000)}.wav"
            temp_file.write_bytes(wav_bytes)
            
            try:
                # Upload file to FAL CDN
                # Note: We tried encode_file() for data URIs but the smart-turn API
                # returns 500 errors when processing them, so we use file upload instead
                audio_url = await fal_client.upload_file_async(str(temp_file))
                self.logger.debug(f"Uploaded audio file for user {user_id}: {audio_url}")

                # Submit to smart-turn model
                handler = await fal_client.submit_async(
                    "fal-ai/smart-turn", arguments={"audio_url": audio_url}
                )

                # Get result
                result = await handler.get()
                await self._process_turn_prediction(user_id, result)
            finally:
                # Clean up temp file
                try:
                    temp_file.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to clean up temp file {temp_file}: {e}")

        except Exception as e:
            self.logger.error(
                f"Error processing audio for user {user_id}: {e}", exc_info=True
            )

    def _create_wav_bytes(self, samples: np.ndarray) -> bytes:
        """
        Create WAV file bytes from audio samples.

        Args:
            samples: numpy array of int16 audio samples

        Returns:
            WAV file bytes
        """
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)  # 16-bit audio
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(samples.tobytes())
        
        # Get the WAV bytes
        wav_bytes = wav_buffer.getvalue()
        
        self.logger.debug(f"Created WAV bytes ({len(samples)} samples, {len(wav_bytes)} bytes)")
        return wav_bytes

    async def _process_turn_prediction(
        self, user_id: str, result: Dict[str, Any]
    ) -> None:
        """
        Process the turn prediction result from FAL API.

        Args:
            user_id: User ID who provided the audio
            result: Result from FAL smart-turn API
        """
        try:
            prediction = result.get("prediction", 0)  # 0 = incomplete, 1 = complete
            probability = result.get("probability", 0.0)

            self.logger.debug(
                f"Turn prediction for {user_id}: {prediction} (prob: {probability:.3f})"
            )

            current_time = time.time()

            # Create event data
            event_data = TurnEventData(
                timestamp=current_time,
                speaker_id=user_id,
                confidence=probability,
                custom={
                    "prediction": prediction,
                    "fal_result": result,
                },
            )

            # Determine if this is a turn completion
            is_complete = prediction == 1 and probability >= self._confidence_threshold

            if is_complete:
                self.logger.info(
                    f"Turn completed detected for user {user_id} (confidence: {probability:.3f})"
                )

                # User finished speaking - emit turn ended
                # Set them as current speaker if they weren't already (in case we missed the start)
                if self._current_speaker != user_id:
                    self._current_speaker = user_id

                self._emit_turn_event(TurnEvent.TURN_ENDED, event_data)
                self._current_speaker = None

            else:
                # Turn is still in progress
                if self._current_speaker != user_id:
                    # New speaker started
                    if self._current_speaker is not None:
                        # Previous speaker ended
                        prev_event_data = TurnEventData(
                            timestamp=current_time,
                            speaker_id=self._current_speaker,
                        )
                        self._emit_turn_event(TurnEvent.TURN_ENDED, prev_event_data)

                    # New speaker started
                    self._current_speaker = user_id
                    self._emit_turn_event(TurnEvent.TURN_STARTED, event_data)
                    self.logger.info(f"Turn started for user {user_id}")

        except Exception as e:
            self.logger.error(
                f"Error processing turn prediction for {user_id}: {e}", exc_info=True
            )


    def stop(self) -> None:
        """Stop turn detection and clean up resources."""
        super().stop()

        # Cancel worker task
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            self.logger.info("Cancelled worker task")

        # Clear the queue
        while not self._processing_queue.empty():
            try:
                self._processing_queue.get_nowait()
                self._processing_queue.task_done()
            except asyncio.QueueEmpty:
                break

        # Clear buffers
        for buffer in self._user_buffers.values():
            buffer.clear()
        self._user_buffers.clear()
        self._user_last_audio.clear()
        self._current_speaker = None

        # Clean up temp directory
        try:
            for file in self._temp_dir.glob("audio_*.wav"):
                file.unlink()
        except Exception as e:
            self.logger.warning(f"Failed to clean up temp files: {e}")

        self.logger.info("Smart Turn detection stopped")

