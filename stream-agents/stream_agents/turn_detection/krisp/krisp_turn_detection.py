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
    
    The Krisp TT model returns probabilities where:
    - Higher values (>0.5) indicate the person should continue speaking
    - Lower values (<0.5) indicate the person has likely finished their turn
    """

    def __init__(
        self,
        model_path: Optional[str] = "./krisp-viva-tt-v1.kef",
        frame_duration_ms: int = 10,  # 10ms frames for Krisp
        prediction_threshold: float = 0.5,
        turn_end_threshold: float = 0.3,  # Lower threshold for detecting turn end
        mini_pause_duration: float = 0.5,
        max_pause_duration: float = 3.0,
        sample_rate: int = 16000,
    ):
        """
        Initialize Krisp turn detection.

        Args:
            model_path: Path to Krisp .kef model file
            frame_duration_ms: Frame duration in milliseconds (10, 15, 20, 30, or 32)
            prediction_threshold: Probability threshold for turn continuation (default 0.5)
            turn_end_threshold: Probability threshold for detecting turn end (default 0.3)
            mini_pause_duration: Duration for mini pause detection
            max_pause_duration: Duration for max pause detection
            sample_rate: Audio sample rate (8000, 16000, 24000, 32000, 44100, or 48000)
        """
        super().__init__(mini_pause_duration, max_pause_duration)

        self.logger = logging.getLogger("KrispTurnDetection")
        self.model_path = model_path
        self.frame_duration_ms = frame_duration_ms
        self.prediction_threshold = prediction_threshold
        self.turn_end_threshold = turn_end_threshold
        self.sample_rate = sample_rate
        self.sample_type = "PCM_16"  # Stream always provides PCM data

        # Validate parameters
        self._validate_parameters()

        # Calculate frame size in samples
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)

        # Audio buffering per user
        self._user_buffers: Dict[str, list] = {}
        self._user_last_audio: Dict[str, float] = {}
        self._current_speaker: Optional[str] = None
        self._user_last_probability: Dict[str, float] = {}

        # Krisp SDK sessions per user
        self._user_sessions: Dict[str, Any] = {}
        
        # Processing state
        self._processing_tasks: Dict[str, asyncio.Task] = {}
        
        # Initialize Krisp SDK
        self._sdk_initialized = False
        self._initialize_krisp_sdk()

    def _validate_parameters(self):
        """Validate Krisp SDK parameters."""
        valid_sample_rates = [8000, 16000, 24000, 32000, 44100, 48000]
        if self.sample_rate not in valid_sample_rates:
            raise ValueError(f"Unsupported sample rate: {self.sample_rate}. Valid rates: {valid_sample_rates}")
        
        valid_frame_durations = [10, 15, 20, 30, 32]
        if self.frame_duration_ms not in valid_frame_durations:
            raise ValueError(f"Unsupported frame duration: {self.frame_duration_ms}ms. Valid durations: {valid_frame_durations}")
        

    def _initialize_krisp_sdk(self):
        """Initialize the Krisp SDK global instance."""
        try:
            def log_callback(log_message, log_level):
                self.logger.debug(f"Krisp [{log_level}]: {log_message}")
            
            krisp_audio.globalInit("", log_callback, krisp_audio.LogLevel.Off)
            self._sdk_initialized = True
            self.logger.info("Krisp SDK initialized successfully")
        except Exception as e:
            self._sdk_initialized = False
            self.logger.error(f"Failed to initialize Krisp SDK: {e}")
            raise

    def _int_to_sample_rate(self, sample_rate: int):
        """Convert integer sample rate to Krisp enum."""
        rates = {
            8000: krisp_audio.SamplingRate.Sr8000Hz,
            16000: krisp_audio.SamplingRate.Sr16000Hz,
            24000: krisp_audio.SamplingRate.Sr24000Hz,
            32000: krisp_audio.SamplingRate.Sr32000Hz,
            44100: krisp_audio.SamplingRate.Sr44100Hz,
            48000: krisp_audio.SamplingRate.Sr48000Hz
        }
        return rates[sample_rate]

    def _int_to_frame_duration(self, frame_dur: int):
        """Convert integer frame duration to Krisp enum."""
        durations = {
            10: krisp_audio.FrameDuration.Fd10ms,
            15: krisp_audio.FrameDuration.Fd15ms,
            20: krisp_audio.FrameDuration.Fd20ms,
            30: krisp_audio.FrameDuration.Fd30ms,
            32: krisp_audio.FrameDuration.Fd32ms
        }
        return durations[frame_dur]

    def _create_user_session(self, user_id: str):
        """Create a Krisp turn-taking session for a user."""
        try:
            self.logger.debug(f"Starting session creation for user {user_id}")
            
            if not self._sdk_initialized:
                raise RuntimeError("Krisp SDK not properly initialized")
            
            if not self.model_path:
                raise ValueError("Model path is required for Krisp turn detection")
            
            self.logger.debug(f"Model path: {self.model_path}")
            
            # Validate model file exists and is readable
            import os
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            if not os.path.isfile(self.model_path):
                raise ValueError(f"Model path is not a file: {self.model_path}")
            if not os.access(self.model_path, os.R_OK):
                raise PermissionError(f"Model file is not readable: {self.model_path}")
            
            file_size = os.path.getsize(self.model_path)
            self.logger.debug(f"Model file exists, size: {file_size} bytes")
            
            # Create model info
            model_info = krisp_audio.ModelInfo()
            model_info.path = self.model_path
            self.logger.debug(f"Created model info with path: {model_info.path}")
            
            # Create turn-taking session configuration
            tt_cfg = krisp_audio.TtSessionConfig()
            sample_rate_enum = self._int_to_sample_rate(self.sample_rate)
            frame_duration_enum = self._int_to_frame_duration(self.frame_duration_ms)
            
            tt_cfg.inputSampleRate = sample_rate_enum
            tt_cfg.inputFrameDuration = frame_duration_enum
            tt_cfg.modelInfo = model_info
            
            self.logger.debug(f"Session config: sample_rate={self.sample_rate}, frame_duration={self.frame_duration_ms}")
            
            # Create PCM_16 turn-taking session (Stream always provides PCM data)
            self.logger.debug("Creating Krisp TtInt16 session...")
            tt_session = krisp_audio.TtInt16.create(tt_cfg)
            self.logger.debug(f"Krisp session created successfully: {tt_session}")
            
            self._user_sessions[user_id] = tt_session
            self._user_last_probability[user_id] = 0.0
            self.logger.info(f"Successfully stored Krisp session for user {user_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to create Krisp session for user {user_id}: {e}", exc_info=True)
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
            self.logger.debug(f"Creating new buffer and session for user {user_id}")
            self._user_buffers[user_id] = []
            try:
                self._create_user_session(user_id)
                self.logger.info(f"Successfully created session for user {user_id}")
            except Exception as e:
                self.logger.error(f"Failed to create session for user {user_id}: {e}", exc_info=True)
                # Don't process audio if session creation failed
                return
        else:
            self.logger.debug(f"Using existing session for user {user_id}")

        # Add audio data to user's buffer
        # Extract audio samples from PcmData object
        self._user_buffers[user_id].extend(audio_data.samples.tolist())

        self._user_last_audio[user_id] = current_time
        buffer_size = len(self._user_buffers[user_id])
        self.logger.debug(f"Buffer size for user {user_id}: {buffer_size}")

        if buffer_size >= self.frame_size:
            # Only start processing if we have a valid session
            if user_id in self._user_sessions and self._user_sessions[user_id] is not None:
                # Start processing task if not already running
                if (
                    user_id not in self._processing_tasks
                    or self._processing_tasks[user_id].done()
                ):
                    self._processing_tasks[user_id] = asyncio.create_task(
                        self._process_user_audio(user_id)
                    )
            else:
                self.logger.warning(f"Cannot process audio for user {user_id}: no valid session")

    async def _process_user_audio(self, user_id: str) -> None:
        """
        Process buffered audio for a specific user through Krisp SDK.

        Args:
            user_id: ID of the user whose audio to process
        """
        try:
            # Process all complete frames available
            while user_id in self._user_buffers and len(self._user_buffers[user_id]) >= self.frame_size:
                # Extract one frame of audio
                audio_frame = self._user_buffers[user_id][:self.frame_size]
                self._user_buffers[user_id] = self._user_buffers[user_id][self.frame_size:]

                self.logger.debug(
                    f"Processing {len(audio_frame)} audio samples for user {user_id}"
                )

                # Process through Krisp SDK
                await self._process_krisp_audio(user_id, audio_frame)

        except Exception as e:
            self.logger.error(
                f"Error processing audio for user {user_id}: {e}", exc_info=True
            )

    async def _process_krisp_audio(self, user_id: str, audio_frame: list):
        """Process audio frame through Krisp turn-taking model."""
        try:
            if user_id not in self._user_sessions:
                self.logger.error(f"No Krisp session found for user {user_id}")
                return
            
            tt_session = self._user_sessions[user_id]
            if tt_session is None:
                self.logger.error(f"Krisp session is None for user {user_id}")
                return
            
            # Convert samples to 16-bit PCM format expected by Krisp
            processed_frame = []
            for sample in audio_frame:
                # Clamp to 16-bit range and convert to int16
                if isinstance(sample, float):
                    # Convert float [-1.0, 1.0] to int16 [-32768, 32767]
                    int_sample = int(sample * 32767)
                else:
                    int_sample = int(sample)
                clamped_sample = max(-32768, min(32767, int_sample))
                processed_frame.append(clamped_sample)

            # Process the audio frame through Krisp
            tt_probability = tt_session.process(processed_frame)
            
            # Handle negative values (Krisp SDK returns -1 when result not ready)
            if tt_probability < 0:
                self.logger.debug(f"Krisp TT result not ready for user {user_id}")
                return
            
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
            previous_probability = self._user_last_probability.get(user_id, 0.0)
            self._user_last_probability[user_id] = probability

            # Create event data
            event_data = TurnEventData(
                timestamp=current_time,
                speaker_id=user_id,
                confidence=probability,
                custom={
                    "krisp_probability": probability,
                    "previous_probability": previous_probability,
                },
            )

            # Determine turn events based on probability changes
            # High probability (>0.5) = person should continue speaking
            # Low probability (<0.3) = person has likely finished their turn
            
            if probability >= self.prediction_threshold:
                # User should be speaking
                if self._current_speaker != user_id:
                    # New speaker started or speaker change
                    if self._current_speaker is not None:
                        # Previous speaker ended
                        prev_event_data = TurnEventData(
                            timestamp=current_time,
                            speaker_id=self._current_speaker,
                            confidence=previous_probability,
                        )
                        self._emit_turn_event(TurnEvent.TURN_ENDED, prev_event_data)

                    # New speaker started
                    self._current_speaker = user_id
                    self._emit_turn_event(TurnEvent.TURN_STARTED, event_data)
                    self.logger.info(f"Turn started for user {user_id} (prob: {probability:.3f})")

            elif probability <= self.turn_end_threshold and self._current_speaker == user_id:
                # Current speaker has likely finished
                self.logger.info(
                    f"Turn completion detected for user {user_id} (prob: {probability:.3f})"
                )
                self._emit_turn_event(TurnEvent.TURN_ENDED, event_data)
                self._current_speaker = None

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
        self._user_last_probability.clear()
        self._current_speaker = None

        # Clean up Krisp sessions
        for user_id, session in self._user_sessions.items():
            try:
                # Let Python garbage collect the session
                session = None
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