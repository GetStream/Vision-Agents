"""
Vogent turn keeping implementation using the Vogent Turn model.

This module provides integration with the Vogent Turn model for multimodal
turn detection combining audio and text context.
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Any

import numpy as np
from getstream.audio.utils import resample_audio
from getstream.video.rtc.track_util import PcmData
from vision_agents.core.utils.utils import to_mono
from vision_agents.core.turn_detection.turn_detection import (
    TurnDetector,
    TurnEvent,
    TurnEventData,
)
from vision_agents.core.edge.types import Participant
from vision_agents.core.agents.conversation import Conversation
from vogent_turn import TurnDetector as VogentTurnDetector


def _resample(samples: np.ndarray) -> np.ndarray:
    """Resample audio from 48 kHz to 16 kHz."""
    return resample_audio(samples, 48000, 16000).astype(np.int16)


class TurnDetection(TurnDetector):
    """
    Turn detection implementation using Vogent Turn model.

    This implementation:
    1. Buffers incoming audio from participants
    2. Processes audio chunks through the Vogent Turn model
    3. Emits turn events based on model predictions
    4. Manages turn state based on predictions
    """

    def start(self) -> None:
        pass

    def __init__(
        self,
        model_name: str = "vogent/Vogent-Turn-80M",
        buffer_duration: float = 2.0,
        confidence_threshold: float = 0.5,
        sample_rate: int = 16000,
        channels: int = 1,
        compile_model: bool = True,
    ):
        """
        Initialize Vogent turn detection.

        Args:
            model_name: HuggingFace model ID for Vogent Turn
            buffer_duration: Duration in seconds to buffer audio before processing
            confidence_threshold: Probability threshold for turn completion
            sample_rate: Audio sample rate (Hz)
            channels: Number of audio channels
            compile_model: Use torch.compile for faster inference
        """
        super().__init__(
            confidence_threshold=confidence_threshold, provider_name="VogentTurnDetection"
        )
        self.logger = logging.getLogger("VogentTurnDetection")
        self.buffer_duration = buffer_duration
        self.sample_rate = sample_rate
        self.channels = channels

        # Initialize Vogent Turn detector
        self.vogent_detector = VogentTurnDetector(
            model_name=model_name,
            compile_model=compile_model,
            warmup=True,
        )

        # Audio buffering per user
        self._user_buffers: Dict[str, bytearray] = {}
        self._user_last_audio: Dict[str, float] = {}
        self._user_conversations: Dict[str, Optional[Conversation]] = {}
        self._current_speaker: Optional[str] = None

        # Processing state
        self._processing_tasks: Dict[str, asyncio.Task] = {}

        self.logger.info(
            f"Initialized Vogent Turn detection (model: {model_name}, "
            f"buffer: {buffer_duration}s, threshold: {confidence_threshold})"
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

    async def process_audio(
        self,
        audio_data: PcmData,
        participant: Participant,
        conversation: Optional[Conversation],
    ) -> None:
        """
        Process incoming audio data for turn detection.

        Args:
            audio_data: PCM audio data from Stream
            participant: Participant that's speaking, includes user data
            conversation: Transcription/ chat history, sometimes useful for turn detection
        """
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
        try:
            samples = _resample(audio_data.samples)
        except Exception as e:
            self.logger.error(f"Failed to resample audio: {e}")
            return

        # Infer number of channels (default to mono)
        num_channels = self._infer_channels(audio_data.format)
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
        self._user_conversations[user_id] = conversation

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
        Process buffered audio for a specific user through Vogent Turn model.

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

            # Convert bytes to float32 numpy array normalized to [-1, 1]
            audio_int16 = np.frombuffer(process_bytes, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0

            self.logger.debug(
                f"Processing {len(audio_float32)} audio samples for user {user_id}"
            )

            # Extract last 2 messages from conversation for context
            prev_line = ""
            curr_line = ""
            conversation = self._user_conversations.get(user_id)
            if conversation and conversation.messages:
                # Get messages for this user, sorted by timestamp
                user_messages = [
                    m for m in conversation.messages 
                    if m.user_id == user_id and m.content
                ]
                if len(user_messages) >= 2:
                    prev_line = user_messages[-2].content
                    curr_line = user_messages[-1].content
                elif len(user_messages) == 1:
                    curr_line = user_messages[-1].content

            # Run prediction in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.vogent_detector.predict,
                audio_float32,
                prev_line,
                curr_line,
                self.sample_rate,
                True,  # return_probs
            )

            await self._process_turn_prediction(user_id, result)

        except Exception as e:
            self.logger.error(
                f"Error processing audio for user {user_id}: {e}", exc_info=True
            )

    async def _process_turn_prediction(
        self, user_id: str, result: Dict[str, Any]
    ) -> None:
        """
        Process the turn prediction result from Vogent Turn model.

        Args:
            user_id: User ID who provided the audio
            result: Result from Vogent Turn prediction
        """
        try:
            is_endpoint = result.get("is_endpoint", False)
            prob_endpoint = result.get("prob_endpoint", 0.0)

            self.logger.debug(
                f"Turn prediction for {user_id}: endpoint={is_endpoint}, "
                f"prob={prob_endpoint:.3f}"
            )

            current_time = time.time()

            # Create event data
            event_data = TurnEventData(
                timestamp=current_time,
                speaker_id=user_id,
                confidence=prob_endpoint,
                custom={
                    "is_endpoint": is_endpoint,
                    "prob_endpoint": prob_endpoint,
                    "prob_continue": result.get("prob_continue", 0.0),
                },
            )

            # Determine if this is a turn completion
            is_complete = is_endpoint and prob_endpoint >= self._confidence_threshold

            if is_complete:
                self.logger.info(
                    f"Turn completed detected for user {user_id} (confidence: {prob_endpoint:.3f})"
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
        self._user_conversations.clear()
        self._current_speaker = None

        self.logger.info("Vogent Turn detection stopped")

