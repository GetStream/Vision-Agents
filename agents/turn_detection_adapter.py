"""
Turn Detection Adapter for Stream Agent Integration.

This module provides an adapter that bridges advanced turn detection implementations
(like Krisp and Fal.ai) with the Stream Agent class, enabling sophisticated
conversation management in video calls.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, Callable, Union
from dataclasses import dataclass
from threading import Lock
import numpy as np

from getstream.models import User
from aiortc import MediaStreamTrack

from turn_detection import (
    BaseTurnDetector,
    TurnEvent,
    TurnEventData,
)


logger = logging.getLogger(__name__)


@dataclass
class AudioBuffer:
    """Manages audio data buffering for a participant."""
    
    user_id: str
    buffer: bytearray = None
    sample_rate: int = 16000
    max_size: int = 32000  # Maximum buffer size in samples
    
    def __post_init__(self):
        """Initialize buffer if not provided."""
        if self.buffer is None:
            self.buffer = bytearray()
    
    def add(self, audio_data: bytes):
        """Add audio data to buffer."""
        self.buffer.extend(audio_data)
        
        # Limit buffer size
        max_bytes = self.max_size * 2  # 16-bit audio = 2 bytes per sample
        if len(self.buffer) > max_bytes:
            self.buffer = self.buffer[-max_bytes:]
    
    def get_numpy(self) -> np.ndarray:
        """Get buffer as numpy array."""
        if len(self.buffer) < 2:
            return np.array([], dtype=np.float32)
        
        # Convert bytes to numpy array
        audio_array = np.frombuffer(self.buffer, dtype=np.int16)
        # Normalize to float32 [-1, 1]
        return audio_array.astype(np.float32) / 32768.0
    
    def clear(self):
        """Clear the buffer."""
        self.buffer = bytearray()


class TurnDetectionAdapter:
    """
    Adapter that bridges advanced turn detectors with the Stream Agent.
    
    This adapter:
    - Wraps BaseTurnDetector implementations (Krisp, Fal, etc.)
    - Converts between Stream audio format and detector format
    - Manages participant tracking
    - Provides both simple (bool) and event-based interfaces
    - Handles audio buffering and processing
    """
    
    def __init__(
        self,
        detector: BaseTurnDetector,
        agent_user_id: Optional[str] = None,
        enable_events: bool = True,
        audio_chunk_ms: int = 100,
    ):
        """
        Initialize the turn detection adapter.
        
        Args:
            detector: The turn detector implementation (Krisp, Fal, etc.)
            agent_user_id: The agent's user ID (to exclude from detection)
            enable_events: Whether to enable event-based turn detection
            audio_chunk_ms: Audio chunk duration in milliseconds
        """
        self.detector = detector
        self.agent_user_id = agent_user_id
        self.enable_events = enable_events
        self.audio_chunk_ms = audio_chunk_ms
        
        # Audio management
        self._audio_buffers: Dict[str, AudioBuffer] = {}
        self._audio_tracks: Dict[str, MediaStreamTrack] = {}
        self._lock = Lock()
        
        # Turn state tracking
        self._current_speaker: Optional[str] = None
        self._agent_should_respond = False
        self._last_turn_end_time: Optional[float] = None
        self._conversation_active = False
        
        # Participant tracking
        self._participants: Dict[str, User] = {}
        
        # Processing tasks
        self._processing_tasks: Dict[str, asyncio.Task] = {}
        
        # Event callbacks
        self._on_agent_turn_callback: Optional[Callable] = None
        self._on_participant_turn_callback: Optional[Callable] = None
        
        # Setup event handlers if events are enabled
        if enable_events:
            self._setup_event_handlers()
        
        logger.info(
            f"TurnDetectionAdapter initialized with {type(detector).__name__}"
        )
    
    def _setup_event_handlers(self):
        """Set up event handlers for the turn detector."""
        
        @self.detector.on(TurnEvent.TURN_STARTED.value)
        def on_turn_started(event_data: TurnEventData):
            self._handle_turn_started(event_data)
        
        @self.detector.on(TurnEvent.TURN_ENDED.value)
        def on_turn_ended(event_data: TurnEventData):
            self._handle_turn_ended(event_data)
        
        @self.detector.on(TurnEvent.SPEECH_STARTED.value)
        def on_speech_started(event_data: TurnEventData):
            logger.debug(f"Speech started: {self._get_speaker_name(event_data.speaker)}")
        
        @self.detector.on(TurnEvent.SPEECH_ENDED.value)
        def on_speech_ended(event_data: TurnEventData):
            logger.debug(f"Speech ended: {self._get_speaker_name(event_data.speaker)}")
        
        @self.detector.on(TurnEvent.MAX_PAUSE_REACHED.value)
        def on_max_pause(event_data: TurnEventData):
            self._handle_max_pause(event_data)
    
    def _get_speaker_name(self, speaker: Optional[User]) -> str:
        """Get display name for a speaker."""
        if not speaker:
            return "Unknown"
        if speaker.custom and "name" in speaker.custom:
            return speaker.custom["name"]
        return speaker.id
    
    def _handle_turn_started(self, event_data: TurnEventData):
        """Handle turn start event."""
        if not event_data.speaker:
            return
        
        speaker_id = event_data.speaker.id
        speaker_name = self._get_speaker_name(event_data.speaker)
        
        with self._lock:
            self._current_speaker = speaker_id
            self._conversation_active = True
            
            # Check if it's a participant speaking (not the agent)
            if speaker_id != self.agent_user_id:
                logger.info(f"🎤 Participant turn started: {speaker_name}")
                
                if self._on_participant_turn_callback:
                    asyncio.create_task(
                        self._safe_callback(
                            self._on_participant_turn_callback,
                            event_data
                        )
                    )
    
    def _handle_turn_ended(self, event_data: TurnEventData):
        """Handle turn end event."""
        if not event_data.speaker:
            return
        
        speaker_id = event_data.speaker.id
        speaker_name = self._get_speaker_name(event_data.speaker)
        
        with self._lock:
            if self._current_speaker == speaker_id:
                self._current_speaker = None
                self._last_turn_end_time = time.time()
                
                # If a participant finished speaking, agent might need to respond
                if speaker_id != self.agent_user_id:
                    logger.info(f"🔚 Participant turn ended: {speaker_name}")
                    self._agent_should_respond = True
    
    def _handle_max_pause(self, event_data: TurnEventData):
        """Handle max pause reached event."""
        with self._lock:
            # Max pause reached - good time for agent to respond
            if self._conversation_active and not self._current_speaker:
                logger.info("⏸️ Max pause reached - agent can respond")
                self._agent_should_respond = True
                
                if self._on_agent_turn_callback:
                    asyncio.create_task(
                        self._safe_callback(
                            self._on_agent_turn_callback,
                            event_data
                        )
                    )
    
    async def _safe_callback(self, callback: Callable, *args, **kwargs):
        """Safely execute a callback."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args, **kwargs)
            else:
                callback(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in callback: {e}")
    
    # --- Stream Agent Interface Methods ---
    
    def detect_turn(self, audio_data: bytes) -> bool:
        """
        Simple turn detection interface for backward compatibility.
        
        Args:
            audio_data: Raw audio bytes from Stream
            
        Returns:
            True if it's the agent's turn to speak
        """
        # This is the simple interface that Agent expects
        # We return the current state of whether agent should respond
        with self._lock:
            should_respond = self._agent_should_respond
            
            # Reset flag after checking
            if should_respond:
                self._agent_should_respond = False
                logger.debug("Agent's turn detected")
            
            return should_respond
    
    async def process_audio(
        self,
        audio_data: Union[bytes, Any],
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Process audio data from a participant.
        
        Args:
            audio_data: Audio data (bytes or PCM object)
            user_id: User ID of the speaker
            metadata: Optional metadata
        """
        if user_id == self.agent_user_id:
            return  # Don't process agent's own audio
        
        try:
            # Extract bytes from PCM object if needed
            if hasattr(audio_data, "data"):
                audio_bytes = audio_data.data
            else:
                audio_bytes = audio_data
            
            # Validate audio data
            if not audio_bytes or not isinstance(audio_bytes, (bytes, bytearray)):
                logger.debug(f"Invalid audio data from {user_id}, skipping")
                return
            
            # Add to buffer
            with self._lock:
                if user_id not in self._audio_buffers:
                    self._audio_buffers[user_id] = AudioBuffer(user_id)
                
                self._audio_buffers[user_id].add(audio_bytes)
                logger.debug(
                    f"Buffered audio for {user_id}: +{len(audio_bytes)} bytes, total ~{len(self._audio_buffers[user_id].buffer)} bytes"
                )
            
            # Process if we have enough data
            await self._process_buffered_audio(user_id)
            
        except Exception as e:
            logger.error(f"Error processing audio for {user_id}: {e}")
    
    async def _process_buffered_audio(self, user_id: str):
        """Process buffered audio for a user."""
        with self._lock:
            if user_id not in self._audio_buffers:
                return
            
            buffer = self._audio_buffers[user_id]
            audio_array = buffer.get_numpy()
            
            # Check if we have enough samples
            # Use default sample rate if detector doesn't have config
            sample_rate = 16000  # Default
            if hasattr(self.detector, 'config') and hasattr(self.detector.config, 'sample_rate'):
                sample_rate = self.detector.config.sample_rate
            
            min_samples = int(sample_rate * self.audio_chunk_ms / 1000)
            if len(audio_array) < min_samples:
                logger.debug(
                    f"Not enough samples yet for {user_id}: {len(audio_array)} < {min_samples}"
                )
                return
            
            # Clear processed audio from buffer
            buffer.clear()
        
        # Process through detector (if it has the right method)
        if hasattr(self.detector, "add_audio_samples"):
            # Preferred path for detectors that support sample ingestion
            logger.debug(
                f"Sending {len(audio_array)} samples to detector for {user_id}"
            )
            self.detector.add_audio_samples(user_id, audio_array)
        elif hasattr(self.detector, "_process_audio_frame"):
            # For Krisp-style detectors
            self.detector._process_audio_frame(user_id, audio_array)
        elif hasattr(self.detector, "_add_to_buffer"):
            # For Fal-style detectors
            # Backward compatibility: just buffer; detector will flush via track handler
            self.detector._add_to_buffer(user_id, audio_array)
    
    async def process_audio_track(self, track: MediaStreamTrack, user_id: str):
        """
        Process a WebRTC audio track.
        
        Args:
            track: WebRTC audio track
            user_id: User ID associated with the track
        """
        if user_id == self.agent_user_id:
            return  # Don't process agent's own audio
        
        # Store track reference
        with self._lock:
            self._audio_tracks[user_id] = track
        
        # If detector supports direct track processing, use it
        if hasattr(self.detector, "process_audio_track"):
            logger.info(f"Processing audio track for {user_id} via detector")
            task = asyncio.create_task(
                self.detector.process_audio_track(track, user_id)
            )
            self._processing_tasks[user_id] = task
        else:
            # Otherwise, process frames manually
            logger.info(f"Processing audio track for {user_id} via adapter")
            task = asyncio.create_task(
                self._process_track_frames(track, user_id)
            )
            self._processing_tasks[user_id] = task
    
    async def _process_track_frames(self, track: MediaStreamTrack, user_id: str):
        """Process frames from an audio track."""
        try:
            while True:
                frame = await track.recv()
                
                # Convert frame to audio data
                audio_data = frame.to_ndarray().tobytes()
                
                # Process through adapter
                await self.process_audio(audio_data, user_id)
                
        except Exception as e:
            if "Connection closed" not in str(e):
                logger.error(f"Error processing track for {user_id}: {e}")
    
    # --- Participant Management ---
    
    def add_participant(self, user: User):
        """
        Add a participant to track.
        
        Args:
            user: User object
        """
        # Get user ID
        user_id = user.id if hasattr(user, 'id') else str(user)
        
        if user_id == self.agent_user_id:
            return  # Don't add agent as participant
        
        with self._lock:
            # Check if already added
            if user_id in self._participants:
                return  # Already tracked
            
            self._participants[user_id] = user
        
        # Add to detector if it supports participant management
        if hasattr(self.detector, "add_participant"):
            self.detector.add_participant(user)
        
        logger.info(f"Added participant: {self._get_speaker_name(user)} ({user_id})")
    
    def remove_participant(self, user_id: str):
        """
        Remove a participant.
        
        Args:
            user_id: User ID to remove
        """
        with self._lock:
            if user_id in self._participants:
                user = self._participants[user_id]
                del self._participants[user_id]
                
                # Clean up audio buffer
                if user_id in self._audio_buffers:
                    del self._audio_buffers[user_id]
                
                # Cancel processing task
                if user_id in self._processing_tasks:
                    self._processing_tasks[user_id].cancel()
                    del self._processing_tasks[user_id]
        
        # Remove from detector
        if hasattr(self.detector, "remove_participant"):
            self.detector.remove_participant(user_id)
        
        logger.info(f"Removed participant: {user_id}")
    
    # --- Lifecycle Management ---
    
    def start(self):
        """Start turn detection."""
        if hasattr(self.detector, "start_detection"):
            self.detector.start_detection()
        
        logger.info("Turn detection started")
    
    def stop(self):
        """Stop turn detection."""
        # Cancel all processing tasks
        for task in self._processing_tasks.values():
            task.cancel()
        self._processing_tasks.clear()
        
        # Stop detector
        if hasattr(self.detector, "stop_detection"):
            self.detector.stop_detection()
        
        # Clear state
        with self._lock:
            self._audio_buffers.clear()
            self._audio_tracks.clear()
            self._participants.clear()
            self._current_speaker = None
            self._agent_should_respond = False
        
        logger.info("Turn detection stopped")
    
    # --- Event Callbacks ---
    
    def on_agent_turn(self, callback: Callable):
        """
        Register callback for when it's the agent's turn.
        
        Args:
            callback: Function to call when agent should respond
        """
        self._on_agent_turn_callback = callback
    
    def on_participant_turn(self, callback: Callable):
        """
        Register callback for when a participant starts speaking.
        
        Args:
            callback: Function to call when participant starts turn
        """
        self._on_participant_turn_callback = callback
    
    # --- Statistics and Insights ---
    
    def get_stats(self) -> Dict[str, Any]:
        """Get turn detection statistics."""
        stats = {
            "current_speaker": self._current_speaker,
            "conversation_active": self._conversation_active,
            "participants": list(self._participants.keys()),
        }
        
        # Add detector-specific stats if available
        if hasattr(self.detector, "get_all_stats"):
            stats["detector_stats"] = self.detector.get_all_stats()
        
        return stats
    
    def get_insights(self) -> Dict[str, Any]:
        """Get conversation insights."""
        insights = {}
        
        # Get insights from detector if available
        if hasattr(self.detector, "get_conversation_insights"):
            insights = self.detector.get_conversation_insights()
        
        return insights