import asyncio
import logging
import time
from typing import Dict, Optional, Any
from stream_agents.turn_detection import BaseTurnDetector, TurnEventData, TurnEvent
from getstream.video.rtc.track_util import PcmData
import krisp_audio

class KrispTurnDetection(BaseTurnDetector):
    """
    Turn detection implementation using Krisp Audio SDK.

    This implementation:
    1. Buffers incoming audio from participants
    2. Processes audio frames through Krisp turn-taking model
    3. Emits turn events based on model predictions
    4. Manages turn state based on model predictions
    """

    def __init__(
            self,
            model_path: Optional[str] = "./krisp-viva-tt-v1.kef",
            buffer_duration: float = 0.01,  # 10ms frames for Krisp
            prediction_threshold: float = 0.5,
            mini_pause_duration: float = 0.5,
            max_pause_duration: float = 3.0,
            sample_rate: int = 16000,
            channels: int = 1,
    ):
        """
        Initialize Krisp turn detection.

        Args:
            model_path: Path to Krisp .kef model file
            buffer_duration: Duration in seconds to buffer audio before processing (10ms for Krisp)
            prediction_threshold: Probability threshold for turn detection
            mini_pause_duration: Duration for mini pause detection
            max_pause_duration: Duration for max pause detection
            sample_rate: Audio sample rate (Hz) - must be 16000 for Krisp
            channels: Number of audio channels (must be 1 for Krisp)
        """
        super().__init__(mini_pause_duration, max_pause_duration)

        self.logger = logging.getLogger("KrispTurnDetection")
        self.model_path = model_path
        self.buffer_duration = buffer_duration
        self.prediction_threshold = prediction_threshold
        self.sample_rate = sample_rate
        self.channels = channels

        # Audio buffering per user
        self._user_buffers: Dict[str, list] = {}
        self._user_last_audio: Dict[str, float] = {}
        self._current_speaker: Optional[str] = None

        # Krisp SDK instances per user
        self._user_sessions: Dict[str, Any] = {}
        
        # Processing state
        self._processing_tasks: Dict[str, asyncio.Task] = {}
        
        # Initialize Krisp SDK
        self._initialize_krisp_sdk()

    def _initialize_krisp_sdk(self):
        """Initialize the Krisp SDK global instance."""
        try:
            # Initialize Krisp SDK with logging callback
            def log_callback(log_message, log_level):
                self.logger.debug(f"Krisp [{log_level}]: {log_message}")
            
            krisp_audio.globalInit("", log_callback, krisp_audio.LogLevel.Off)
            self.logger.info("Krisp SDK initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Krisp SDK: {e}")
            raise

    def _create_user_session(self, user_id: str):
        """Create a Krisp turn-taking session for a user."""
        try:
            if not self.model_path:
                raise ValueError("Model path is required for Krisp turn detection")
            
            # Create model info
            model_info = krisp_audio.ModelInfo()
            model_info.path = self.model_path
            
            # Create turn-taking session configuration
            tt_cfg = krisp_audio.TtSessionConfig()
            tt_cfg.inputSampleRate = krisp_audio.SamplingRate.Sr16000Hz
            tt_cfg.inputFrameDuration = krisp_audio.FrameDuration.Fd10ms
            tt_cfg.modelInfo = model_info
            
            # Create turn-taking session
            tt_session = krisp_audio.TtFloat.create(tt_cfg)
            
            self._user_sessions[user_id] = tt_session
            self.logger.debug(f"Created Krisp session for user {user_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to create Krisp session for user {user_id}: {e}")
            raise

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

        # Initialize buffer and session for new user
        if user_id not in self._user_buffers:
            self._user_buffers[user_id] = []
            self._create_user_session(user_id)
            self.logger.debug(f"Initialized audio buffer and Krisp session for user {user_id}")

        # Add audio data to user's buffer
        # Extract audio samples from PcmData object
        if hasattr(audio_data, "samples"):
            # PcmData NamedTuple - extract samples (numpy array)
            samples = audio_data.samples
            if hasattr(samples, "tolist"):
                # Convert numpy array to list for extending buffer
                audio_samples = samples.tolist()
            else:
                audio_samples = list(samples)
            self._user_buffers[user_id].extend(audio_samples)
        elif hasattr(audio_data, "data"):
            # Fallback for data attribute
            self._user_buffers[user_id].extend(audio_data.data)
        else:
            # Assume it's already iterable (bytes, list, etc.)
            self._user_buffers[user_id].extend(audio_data)

        self._user_last_audio[user_id] = current_time

        # Check if we have enough audio to process (10ms frame for Krisp)
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
        Process buffered audio for a specific user through Krisp SDK.

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

            # Process through Krisp SDK
            await self._process_krisp_audio(user_id, process_samples)

        except Exception as e:
            self.logger.error(
                f"Error processing audio for user {user_id}: {e}", exc_info=True
            )

    async def _process_krisp_audio(self, user_id: str, samples: list):
        """Process audio samples through Krisp turn-taking model."""
        try:
            if user_id not in self._user_sessions:
                self.logger.error(f"No Krisp session found for user {user_id}")
                return

            tt_session = self._user_sessions[user_id]
            
            # Convert samples to the format expected by Krisp
            # Krisp expects 16-bit PCM samples
            if isinstance(samples[0], (int, float)):
                # Ensure samples are in the correct range for 16-bit audio
                audio_frame = []
                for sample in samples:
                    # Clamp to 16-bit range and convert to int16
                    clamped_sample = max(-32768, min(32767, int(sample)))
                    audio_frame.append(clamped_sample)
            else:
                audio_frame = samples

            # Process the audio frame through Krisp
            tt_probability = tt_session.process(audio_frame)
            
            # tt_probability is in the [0, 1] range
            self.logger.debug(
                f"Krisp turn probability for {user_id}: {tt_probability:.3f}"
            )

            # Process the turn prediction
            await self._process_turn_prediction(user_id, tt_probability)

        except Exception as e:
            self.logger.error(
                f"Error processing audio through Krisp for user {user_id}: {e}", exc_info=True
            )

    async def _process_turn_prediction(
            self, user_id: str, probability: float
    ) -> None:
        """
        Process the turn prediction result from Krisp SDK.

        Args:
            user_id: User ID who provided the audio
            probability: Turn probability from Krisp (0.0 to 1.0)
        """
        try:
            current_time = time.time()

            # Create event data
            event_data = TurnEventData(
                timestamp=current_time,
                speaker_id=user_id,
                confidence=probability,
                custom={
                    "krisp_probability": probability,
                },
            )

            # Determine if this indicates a turn completion
            is_complete = probability >= self.prediction_threshold

            if is_complete:
                self.logger.info(
                    f"Turn completion detected for user {user_id} (confidence: {probability:.3f})"
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

    def start_detection(self) -> None:
        """Start turn detection."""
        super().start_detection()
        self.logger.info("Krisp turn detection started")

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

        # Clean up Krisp sessions
        for user_id, session in self._user_sessions.items():
            try:
                session = None  # Let Python garbage collect the session
                self.logger.debug(f"Cleaned up Krisp session for user {user_id}")
            except Exception as e:
                self.logger.warning(f"Failed to clean up Krisp session for user {user_id}: {e}")
        
        self._user_sessions.clear()

        # Clean up Krisp SDK global instance
        try:
            krisp_audio.globalDestroy()
            self.logger.info("Krisp SDK global instance destroyed")
        except Exception as e:
            self.logger.warning(f"Failed to destroy Krisp SDK global instance: {e}")

        self.logger.info("Krisp turn detection stopped")




