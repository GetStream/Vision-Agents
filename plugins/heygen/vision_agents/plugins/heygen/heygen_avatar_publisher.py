import asyncio
import logging
from typing import Optional, Any, Tuple

from vision_agents.core.processors.base_processor import (
    AudioVideoProcessor,
    VideoPublisherMixin,
)

from .heygen_rtc_manager import HeyGenRTCManager
from .heygen_video_track import HeyGenVideoTrack

logger = logging.getLogger(__name__)


class AvatarPublisher(AudioVideoProcessor, VideoPublisherMixin):
    """HeyGen avatar video publisher.
    
    Publishes video of a HeyGen avatar that lip-syncs to audio input.
    Can be used as a processor in the Vision Agents framework to add
    realistic avatar video to AI agents.
    
    Example:
        agent = Agent(
            edge=getstream.Edge(),
            agent_user=User(name="Avatar AI"),
            instructions="Be helpful and friendly",
            llm=gemini.LLM("gemini-2.0-flash"),
            tts=cartesia.TTS(),
            stt=deepgram.STT(),
            processors=[
                heygen.AvatarPublisher(
                    avatar_id="default",
                    quality="high"
                )
            ]
        )
    """

    def __init__(
        self,
        avatar_id: str = "default",
        quality: str = "high",
        resolution: Tuple[int, int] = (1920, 1080),
        api_key: Optional[str] = None,
        interval: int = 0,
        **kwargs,
    ):
        """Initialize the HeyGen avatar publisher.
        
        Args:
            avatar_id: HeyGen avatar ID to use for streaming.
            quality: Video quality ("low", "medium", "high").
            resolution: Output video resolution (width, height).
            api_key: HeyGen API key. Uses HEYGEN_API_KEY env var if not provided.
            interval: Processing interval (not used, kept for compatibility).
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(
            interval=interval,
            receive_audio=True,  # Receive audio to forward to HeyGen for lip-sync
            receive_video=False,
            **kwargs
        )
        
        self.avatar_id = avatar_id
        self.quality = quality
        self.resolution = resolution
        self.api_key = api_key
        
        # WebRTC manager for HeyGen connection
        self.rtc_manager = HeyGenRTCManager(
            avatar_id=avatar_id,
            quality=quality,
            api_key=api_key,
        )
        
        # Video track for publishing avatar frames
        self._video_track = HeyGenVideoTrack(
            width=resolution[0],
            height=resolution[1],
        )
        
        # Connection state
        self._connected = False
        self._connection_task: Optional[asyncio.Task] = None
        self._audio_track_set = False
        self._agent = None  # Will be set by the agent
        
        # Create a custom audio track for HeyGen that we can write to
        from .heygen_audio_track import HeyGenAudioTrack
        self._heygen_audio_track = HeyGenAudioTrack(sample_rate=24000)
        
        logger.info(
            f"🎭 HeyGen AvatarPublisher initialized "
            f"(avatar: {avatar_id}, quality: {quality}, resolution: {resolution})"
        )
    
    def set_agent(self, agent: Any) -> None:
        """Set the agent reference for event subscription.
        
        This is called by the agent when the processor is attached.
        
        Args:
            agent: The agent instance.
        """
        self._agent = agent
        logger.info("🔗 Agent reference set for HeyGen avatar publisher")

    async def _connect_to_heygen(self) -> None:
        """Establish connection to HeyGen and start receiving video."""
        try:
            # Set up video callback before connecting
            self.rtc_manager.set_video_callback(self._on_video_track)
            
            # Connect to HeyGen
            await self.rtc_manager.connect()
            
            self._connected = True
            logger.info("✅ Connected to HeyGen, avatar streaming active")
            
            # Subscribe to audio output events from the LLM for lip-sync
            self._subscribe_to_audio_events()
        
        except Exception as e:
            logger.error(f"❌ Failed to connect to HeyGen: {e}")
            self._connected = False
            raise
    
    def _subscribe_to_audio_events(self) -> None:
        """Subscribe to audio output events from the LLM."""
        try:
            # Import the event type
            from vision_agents.core.llm.events import RealtimeAudioOutputEvent
            
            # Get the agent's event manager
            # Note: This will be set when the processor is attached to an agent
            if hasattr(self, '_agent') and self._agent:
                @self._agent.events.subscribe
                async def on_audio_output(event: RealtimeAudioOutputEvent):
                    logger.debug(f"📢 Received audio output event: {len(event.audio_data)} bytes at {event.sample_rate}Hz")
                    await self._on_audio_output(event.audio_data, event.sample_rate)
                logger.info("🎧 Subscribed to LLM audio output events for lip-sync")
                
                # Also log what events are registered
                logger.info(f"   Event manager has {len(self._agent.events._handlers)} event handlers")
            else:
                logger.warning("⚠️ Cannot subscribe to audio events - no agent attached yet")
        except Exception as e:
            logger.error(f"Failed to subscribe to audio events: {e}")
            import traceback
            logger.error(traceback.format_exc())

    async def _on_video_track(self, track: Any) -> None:
        """Callback when video track is received from HeyGen.
        
        Args:
            track: Incoming video track from HeyGen's WebRTC connection.
        """
        logger.info("📹 Received video track from HeyGen, starting frame forwarding")
        await self._video_track.start_receiving(track)

    async def _setup_audio_forwarding(self) -> None:
        """Set up audio forwarding from agent to HeyGen for lip-sync."""
        if self._audio_track_set:
            return  # Already set up
        
        logger.info("🎤 Setting up audio forwarding to HeyGen for lip-sync")
        
        # Wait for HeyGen connection
        if not self._connected:
            if self._connection_task:
                try:
                    await asyncio.wait_for(self._connection_task, timeout=10.0)
                except asyncio.TimeoutError:
                    logger.error("Timeout waiting for HeyGen connection")
                    return
            else:
                logger.error("HeyGen connection not started")
                return
        
        # Set our custom audio track on the HeyGen sender
        await self.rtc_manager.send_audio_track(self._heygen_audio_track)
        self._audio_track_set = True
        logger.info("✅ Audio track set up for HeyGen lip-sync")

    async def _on_audio_output(self, audio_data: bytes, sample_rate: int) -> None:
        """Handle audio output from the LLM and forward to HeyGen.
        
        Args:
            audio_data: Raw PCM audio data from the LLM.
            sample_rate: Sample rate of the audio data.
        """
        logger.debug(f"🎵 _on_audio_output called: {len(audio_data)} bytes at {sample_rate}Hz")
        
        if not self._audio_track_set:
            # Set up audio forwarding on first audio output
            logger.info("🔧 Setting up audio forwarding on first audio output")
            await self._setup_audio_forwarding()
        
        # Write audio data to our custom track for HeyGen
        logger.info(f"✍️ Writing {len(audio_data)} bytes to HeyGen audio track")
        self._heygen_audio_track.write_audio(audio_data)

    def set_agent_audio_track(self, audio_track: Any) -> None:
        """Set the agent's audio track for forwarding to HeyGen.
        
        DEPRECATED: This method is no longer needed. Audio is now forwarded
        via event listening instead of track sharing.
        
        Args:
            audio_track: The agent's audio output track (unused).
        """
        logger.warning("set_agent_audio_track is deprecated - audio forwarding is automatic via events")

    def publish_video_track(self):
        """Publish the HeyGen avatar video track.
        
        This method is called by the Agent to get the video track
        for publishing to the call.
        
        Returns:
            HeyGenVideoTrack instance for streaming avatar video.
        """
        # Start connection if not already connected
        if not self._connected and not self._connection_task:
            self._connection_task = asyncio.create_task(self._connect_to_heygen())
        
        logger.info("🎥 Publishing HeyGen avatar video track")
        return self._video_track

    def state(self) -> dict:
        """Get current state of the avatar publisher.
        
        Returns:
            Dictionary containing current state information.
        """
        return {
            "avatar_id": self.avatar_id,
            "quality": self.quality,
            "resolution": self.resolution,
            "connected": self._connected,
            "rtc_connected": self.rtc_manager.is_connected,
        }

    async def close(self) -> None:
        """Clean up resources and close connections."""
        logger.info("🔌 Closing HeyGen avatar publisher")
        
        # Stop video track
        if self._video_track:
            self._video_track.stop()
        
        # Close RTC connection
        if self.rtc_manager:
            await self.rtc_manager.close()
        
        # Cancel connection task if running
        if self._connection_task:
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass
        
        self._connected = False
        logger.info("✅ HeyGen avatar publisher closed")

