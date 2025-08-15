"""
FAL Smart-Turn implementation for turn detection using the smart-turn AI model.

This module provides integration with the FAL AI smart-turn model to detect
when a speaker has completed their turn in a conversation.
"""

import asyncio
import logging
import tempfile
import time
import wave
from typing import Dict, Optional, Any
from pathlib import Path

import fal_client
from getstream.video.rtc.track_util import PcmData

from .turn_detection import BaseTurnDetector, TurnEvent, TurnEventData


class FalTurnDetection(BaseTurnDetector):
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
        prediction_threshold: float = 0.5,
        mini_pause_duration: float = 0.5,
        max_pause_duration: float = 3.0,
        sample_rate: int = 16000,
        channels: int = 1,
    ):
        """
        Initialize FAL turn detection.

        Args:
            api_key: FAL API key (if None, uses FAL_KEY env var)
            buffer_duration: Duration in seconds to buffer audio before processing
            prediction_threshold: Probability threshold for "complete" predictions
            mini_pause_duration: Duration for mini pause detection
            max_pause_duration: Duration for max pause detection
            sample_rate: Audio sample rate (Hz)
            channels: Number of audio channels
        """
        super().__init__(mini_pause_duration, max_pause_duration)

        self.logger = logging.getLogger("FalTurnDetection")
        self.api_key = api_key
        self.buffer_duration = buffer_duration
        self.prediction_threshold = prediction_threshold
        self.sample_rate = sample_rate
        self.channels = channels

        # Audio buffering per user
        self._user_buffers: Dict[str, list] = {}
        self._user_last_audio: Dict[str, float] = {}
        self._current_speaker: Optional[str] = None

        # Processing state
        self._processing_tasks: Dict[str, asyncio.Task] = {}
        self._temp_dir = Path(tempfile.gettempdir()) / "fal_turn_detection"
        self._temp_dir.mkdir(exist_ok=True)

        # Configure FAL client
        if self.api_key:
            fal_client.api_key = self.api_key

        self.logger.info(
            f"Initialized FAL turn detection (buffer: {buffer_duration}s, threshold: {prediction_threshold})"
        )

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

        current_time = time.time()

        # Initialize buffer for new user
        if user_id not in self._user_buffers:
            self._user_buffers[user_id] = []
            self.logger.debug(f"Initialized audio buffer for user {user_id}")

        # Add audio data to user's buffer
        self._user_buffers[user_id].extend(audio_data.data)
        self._user_last_audio[user_id] = current_time

        # Check if we have enough audio to process
        buffer_size = len(self._user_buffers[user_id])
        required_samples = int(self.buffer_duration * self.sample_rate)

        if buffer_size >= required_samples:
            # Start processing task if not already running
            if (
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

            audio_samples = self._user_buffers[user_id].copy()
            required_samples = int(self.buffer_duration * self.sample_rate)

            if len(audio_samples) < required_samples:
                return

            # Take the required samples and clear processed portion
            process_samples = audio_samples[:required_samples]
            self._user_buffers[user_id] = audio_samples[required_samples:]

            self.logger.debug(
                f"Processing {len(process_samples)} audio samples for user {user_id}"
            )

            # Create temporary audio file
            temp_file = await self._create_audio_file(process_samples, user_id)

            try:
                # Upload to FAL
                audio_url = await fal_client.upload_file_async(str(temp_file))
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

        # Convert samples to bytes if needed
        if isinstance(samples[0], int):
            # Convert int16 samples to bytes
            audio_bytes = bytearray()
            for sample in samples:
                audio_bytes.extend(sample.to_bytes(2, byteorder="little", signed=True))
        else:
            audio_bytes = bytes(samples)

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
                speaker=None,  # We could create a User object here if needed
                confidence=probability,
                custom={
                    "user_id": user_id,
                    "prediction": prediction,
                    "fal_result": result,
                },
            )

            # Determine if this is a turn completion
            is_complete = prediction == 1 and probability >= self.prediction_threshold

            if is_complete:
                self.logger.info(
                    f"Turn completed detected for user {user_id} (confidence: {probability:.3f})"
                )

                # If this user was speaking, emit turn ended
                if self._current_speaker == user_id:
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
                            custom={"user_id": self._current_speaker},
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

    def start_detection(self) -> None:
        """Start turn detection."""
        super().start_detection()
        self.logger.info("FAL turn detection started")

    def stop_detection(self) -> None:
        """Stop turn detection and clean up."""
        super().stop_detection()

        # Cancel any running processing tasks
        for task in self._processing_tasks.values():
            if not task.done():
                task.cancel()
        self._processing_tasks.clear()

        # Clear buffers
        self._user_buffers.clear()
        self._user_last_audio.clear()
        self._current_speaker = None

        # Clean up temp directory
        try:
            for file in self._temp_dir.glob("audio_*.wav"):
                file.unlink()
        except Exception as e:
            self.logger.warning(f"Failed to clean up temp files: {e}")

        self.logger.info("FAL turn detection stopped")
