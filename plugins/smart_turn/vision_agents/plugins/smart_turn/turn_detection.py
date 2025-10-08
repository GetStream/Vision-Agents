"""
Smart Turn detection implementation using the FAL AI smart-turn model.

This module provides integration with the FAL AI smart-turn model to detect
when a speaker has completed their turn in a conversation.
"""

import asyncio
import os
import logging
import tempfile
import time
import wave
from typing import Dict, Optional, Any
from pathlib import Path

import fal_client
import numpy as np
from getstream.audio.utils import resample_audio
from getstream.video.rtc.track_util import PcmData
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
    Turn detection implementation using FAL AI's smart-turn model.

    This implementation:
    1. Buffers incoming audio from participants
    2. Periodically uploads audio chunks to FAL API
    3. Processes smart-turn predictions to emit turn events
    4. Manages turn state based on model predictions
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        buffer_duration: float = 2.0,
        confidence_threshold: float = 0.5,
        sample_rate: int = 16000,
        channels: int = 1,
    ):
        """
        Initialize Smart Turn detection.

        Args:
            api_key: FAL API key (if None, uses FAL_KEY env var)
            buffer_duration: Duration in seconds to buffer audio before processing
            confidence_threshold: Probability threshold for "complete" predictions
            sample_rate: Audio sample rate (Hz)
            channels: Number of audio channels
        """

        super().__init__(
            confidence_threshold=confidence_threshold, provider_name="SmartTurnDetection"
        )
        self.logger = logging.getLogger("SmartTurnDetection")
        self.api_key = api_key
        self.buffer_duration = buffer_duration
        self.sample_rate = sample_rate
        self.channels = channels

        # Audio buffering per user
        self._user_buffers: Dict[str, bytearray] = {}
        self._user_last_audio: Dict[str, float] = {}
        self._current_speaker: Optional[str] = None

        # Processing state
        self._processing_tasks: Dict[str, asyncio.Task] = {}
        self._temp_dir = Path(tempfile.gettempdir()) / "smart_turn_detection"
        self._temp_dir.mkdir(exist_ok=True)

        # Configure FAL client
        if self.api_key:
            os.environ["FAL_KEY"] = self.api_key

        self.logger.info(
            f"Initialized Smart Turn detection (buffer: {buffer_duration}s, threshold: {confidence_threshold})"
        )

    def _infer_channels(self, format_str: str) -> int:
        """Infer number of channels from PcmData format string."""
        format_str = format_str.lower()
        if "stereo" in format_str:
            return 2
        elif any(f in format_str for f in ["mono", "s16", "int16", "pcm_s16le"]):
            return 1
        else:
            self.logger.warning(
                f"Unknown format string: {format_str}. Assuming mono."
            )
            return 1

    def is_detecting(self) -> bool:
        """Check if turn detection is currently active."""
        return self._is_detecting

    async def process_audio(
        self,
        audio_data: PcmData,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Process incoming audio data for turn detection.

        Args:
            audio_data: PCM audio data from Stream
            user_id: ID of the user speaking
            metadata: Optional metadata about the audio
        """
        if not self.is_detecting():
            return

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
        try:
            samples = _resample(audio_data.samples)
        except Exception as e:
            self.logger.error(f"Failed to resample audio: {e}")
            return

        # Infer number of channels (default to mono)
        num_channels = (
            metadata.get("channels", self._infer_channels(audio_data.format))
            if metadata
            else self._infer_channels(audio_data.format)
        )
        if num_channels != 1:
            self.logger.debug(f"Converting {num_channels}-channel audio to mono")
            try:
                samples = to_mono(samples, num_channels)
            except ValueError as e:
                self.logger.error(f"Failed to convert to mono: {e}")
                return

        # Initialize buffer for new user
        self._user_buffers.setdefault(user_id, bytearray())
        self._user_last_audio[user_id] = time.time()

        # Convert samples to bytes and append to buffer
        self._user_buffers[user_id].extend(samples.tobytes())

        # Process audio if buffer is large enough and no task is running
        buffer_size = len(self._user_buffers[user_id])
        required_bytes = int(
            self.buffer_duration * self.sample_rate * 2
        )  # 2 bytes per int16 sample
        if buffer_size >= required_bytes and (
            user_id not in self._processing_tasks
            or self._processing_tasks[user_id].done()
        ):
            self._processing_tasks[user_id] = asyncio.create_task(
                self._process_user_audio(user_id)
            )

    async def _process_user_audio(self, user_id: str) -> None:
        """
        Process buffered audio for a specific user through FAL API.

        Args:
            user_id: ID of the user whose audio to process
        """
        try:
            # Extract audio buffer
            if user_id not in self._user_buffers:
                return

            audio_buffer = self._user_buffers[user_id]
            required_bytes = int(
                self.buffer_duration * self.sample_rate * 2
            )  # 2 bytes per int16 sample

            if len(audio_buffer) < required_bytes:
                return

            # Take the required bytes and clear processed portion
            process_bytes = bytes(audio_buffer[:required_bytes])
            del audio_buffer[:required_bytes]

            # Convert bytes back to samples for WAV creation
            process_samples = np.frombuffer(process_bytes, dtype=np.int16).tolist()

            self.logger.debug(
                f"Processing {len(process_samples)} audio samples for user {user_id}"
            )

            # Create temporary audio file
            temp_file = await self._create_audio_file(process_samples, user_id)

            try:
                # Upload to FAL
                audio_url = await fal_client.upload_file_async(temp_file)
                self.logger.debug(
                    f"Uploaded audio file for user {user_id}: {audio_url}"
                )

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
                    self.logger.warning(
                        f"Failed to clean up temp file {temp_file}: {e}"
                    )

        except Exception as e:
            self.logger.error(
                f"Error processing audio for user {user_id}: {e}", exc_info=True
            )

    async def _create_audio_file(self, samples: list, user_id: str) -> Path:
        """
        Create a temporary WAV file from audio samples.

        Args:
            samples: List of audio samples
            user_id: User ID for unique filename

        Returns:
            Path to the created audio file
        """
        timestamp = int(time.time() * 1000)
        filename = f"audio_{user_id}_{timestamp}.wav"
        filepath = self._temp_dir / filename

        # Convert samples to bytes - samples are already a list of int16 values
        audio_bytes_array = bytearray()
        for sample in samples:
            audio_bytes_array.extend(
                sample.to_bytes(2, byteorder="little", signed=True)
            )
        audio_bytes = bytes(audio_bytes_array)

        # Create WAV file
        with wave.open(str(filepath), "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)  # 16-bit audio
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_bytes)

        self.logger.debug(f"Created audio file: {filepath} ({len(samples)} samples)")
        return filepath

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

    def start(self) -> None:
        """Start turn detection."""
        if self._is_detecting:
            return
        self._is_detecting = True
        self.logger.info("Smart Turn detection started")

    def stop(self) -> None:
        """Stop turn detection and clean up."""
        if not self._is_detecting:
            return
        self._is_detecting = False

        # Cancel any running processing tasks
        for task in self._processing_tasks.values():
            if not task.done():
                task.cancel()
        self._processing_tasks.clear()

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

