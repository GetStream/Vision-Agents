import asyncio
import logging
from typing import Optional, Any, Tuple

import numpy as np
from getstream.video.rtc import audio_track

from vision_agents.core.processors.base_processor import (
    AudioVideoProcessor,
    VideoPublisherMixin,
    AudioPublisherMixin,
)

from .heygen_rtc_manager import HeyGenRTCManager
from .heygen_video_track import HeyGenVideoTrack

logger = logging.getLogger(__name__)


class AvatarPublisher(AudioVideoProcessor, VideoPublisherMixin, AudioPublisherMixin):
    """HeyGen avatar video and audio publisher.
    
    Publishes video of a HeyGen avatar that lip-syncs based on LLM text output.
    Can be used as a processor in the Vision Agents framework to add
    realistic avatar video to AI agents.
    
    HeyGen handles TTS internally, so no separate TTS is needed.
    
    Example:
        agent = Agent(
            edge=getstream.Edge(),
            agent_user=User(name="Avatar AI"),
            instructions="Be helpful and friendly",
            llm=gemini.LLM("gemini-2.0-flash"),
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
        mute_llm_audio: bool = True,
        **kwargs,
    ):
        """Initialize the HeyGen avatar publisher.
        
        Args:
            avatar_id: HeyGen avatar ID to use for streaming.
            quality: Video quality ("low", "medium", "high").
            resolution: Output video resolution (width, height).
            api_key: HeyGen API key. Uses HEYGEN_API_KEY env var if not provided.
            interval: Processing interval (not used, kept for compatibility).
            mute_llm_audio: If True, mutes the Realtime LLM's audio output so only
                HeyGen's video (with audio) is heard. Default: True.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(
            interval=interval,
            receive_audio=False,  # We send text to HeyGen, not audio
            receive_video=False,
            **kwargs
        )
        
        self.avatar_id = avatar_id
        self.quality = quality
        self.resolution = resolution
        self.api_key = api_key
        self.mute_llm_audio = mute_llm_audio
        
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
        
        # Audio track for publishing HeyGen's audio
        # Create it immediately so the agent can detect it during initialization
        self._audio_track = audio_track.AudioStreamTrack(
            framerate=48000, stereo=True
        )
        
        # Connection state
        self._connected = False
        self._connection_task: Optional[asyncio.Task] = None
        self._agent = None  # Will be set by the agent
        
        # Text buffer for accumulating LLM response chunks before sending to HeyGen
        self._text_buffer = ""
        self._current_response_id: Optional[str] = None
        self._sent_texts: set = set()  # Track sent texts to avoid duplicates
        
        # Audio forwarding state (for selective muting of Realtime LLM audio)
        self._forwarding_audio = False
        
        logger.info(
            f"🎭 HeyGen AvatarPublisher initialized "
            f"(avatar: {avatar_id}, quality: {quality}, resolution: {resolution})"
        )
    
    def publish_audio_track(self):
        """Return the audio track for publishing HeyGen's audio.
        
        This method is called by the Agent to get the audio track that will
        be published to the call. HeyGen's audio will be forwarded to this track.
        """
        return self._audio_track
    
    def set_agent(self, agent: Any) -> None:
        """Set the agent reference for event subscription.
        
        This is called by the agent when the processor is attached.
        
        Args:
            agent: The agent instance.
        """
        self._agent = agent
        logger.info("🔗 Agent reference set for HeyGen avatar publisher")
        
        # Mute the Realtime LLM's audio if requested
        if self.mute_llm_audio:
            self._mute_realtime_llm_audio()
        
        # Subscribe to text events immediately when agent is set
        self._subscribe_to_text_events()

    async def _connect_to_heygen(self) -> None:
        """Establish connection to HeyGen and start receiving video and audio."""
        try:
            # Set up video and audio callbacks before connecting
            self.rtc_manager.set_video_callback(self._on_video_track)
            self.rtc_manager.set_audio_callback(self._on_audio_track)
            
            # Connect to HeyGen
            await self.rtc_manager.connect()
            
            self._connected = True
            logger.info("✅ Connected to HeyGen, avatar streaming active")
        
        except Exception as e:
            logger.error(f"❌ Failed to connect to HeyGen: {e}")
            self._connected = False
            raise
    
    def _subscribe_to_text_events(self) -> None:
        """Subscribe to text output events from the LLM.
        
        HeyGen requires text input (not audio) for proper lip-sync.
        We listen to the LLM's text output and send it to HeyGen's task API.
        """
        try:
            # Import the event types
            from vision_agents.core.llm.events import (
                LLMResponseChunkEvent,
                LLMResponseCompletedEvent,
                RealtimeAgentSpeechTranscriptionEvent,
            )
            
            # Get the LLM's event manager (events are emitted by the LLM, not the agent)
            if hasattr(self, '_agent') and self._agent and hasattr(self._agent, 'llm'):
                @self._agent.llm.events.subscribe
                async def on_text_chunk(event: LLMResponseChunkEvent):
                    """Handle streaming text chunks from the LLM."""
                    logger.debug(f"📝 HeyGen received text chunk: delta='{event.delta}'")
                    if event.delta:
                        await self._on_text_chunk(event.delta, event.item_id)
                
                @self._agent.llm.events.subscribe
                async def on_text_complete(event: LLMResponseCompletedEvent):
                    """Handle end of LLM response - send any remaining buffered text."""
                    # Send any remaining buffered text
                    if self._text_buffer.strip():
                        text_to_send = self._text_buffer.strip()
                        if text_to_send not in self._sent_texts:
                            await self._send_text_to_heygen(text_to_send)
                            self._sent_texts.add(text_to_send)
                        self._text_buffer = ""
                    # Reset for next response
                    self._current_response_id = None
                    self._sent_texts.clear()
                
                @self._agent.llm.events.subscribe
                async def on_agent_speech(event: RealtimeAgentSpeechTranscriptionEvent):
                    """Handle agent speech transcription from Realtime LLMs.
                    
                    This is the primary path for Gemini Realtime which transcribes
                    the agent's speech output as text.
                    """
                    logger.debug(f"📝 HeyGen received agent speech: text='{event.text}'")
                    if event.text:
                        # Send directly to HeyGen - this is the complete utterance
                        await self._send_text_to_heygen(event.text)
                
                logger.info("📝 Subscribed to LLM text output events for HeyGen lip-sync")
            else:
                logger.warning("⚠️ Cannot subscribe to text events - no agent or LLM attached yet")
        except Exception as e:
            logger.error(f"Failed to subscribe to text events: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _mute_realtime_llm_audio(self) -> None:
        """Mute the Realtime LLM's audio output.
        
        When using HeyGen, we want HeyGen to handle all audio (with lip-sync),
        so we mute the LLM's native audio output to avoid duplicated/overlapping audio.
        
        This works by intercepting writes to the LLM's output_track and only blocking
        writes that come from the LLM itself (not from HeyGen forwarding).
        """
        try:
            from vision_agents.core.llm.realtime import Realtime
            
            if not hasattr(self, '_agent') or not self._agent:
                logger.warning("⚠️ Cannot mute LLM audio - no agent set")
                return
                
            if not hasattr(self._agent, 'llm') or not isinstance(self._agent.llm, Realtime):
                logger.info("ℹ️ LLM is not a Realtime LLM - no audio to mute")
                return
            
            # Store the original write method
            original_write = self._agent.llm.output_track.write
            
            # Create a selective write method
            async def selective_write(audio_data: bytes) -> None:
                """Only allow writes from HeyGen forwarding, block LLM writes."""
                if self._forwarding_audio:
                    # This is from HeyGen - allow it
                    await original_write(audio_data)
                # else: This is from the Realtime LLM - block it
            
            # Replace the write method
            self._agent.llm.output_track.write = selective_write
            
            logger.info("🔇 Muted Realtime LLM audio output (HeyGen will provide audio)")
            
        except Exception as e:
            logger.error(f"Failed to mute LLM audio: {e}")
            import traceback
            logger.error(traceback.format_exc())

    async def _on_video_track(self, track: Any) -> None:
        """Callback when video track is received from HeyGen.
        
        Args:
            track: Incoming video track from HeyGen's WebRTC connection.
        """
        logger.info("📹 Received video track from HeyGen, starting frame forwarding")
        await self._video_track.start_receiving(track)

    async def _on_audio_track(self, track: Any) -> None:
        """Callback when audio track is received from HeyGen.
        
        HeyGen provides audio with lip-synced TTS. We forward this audio
        to the agent's audio track so it gets published to the call.
        
        Args:
            track: Incoming audio track from HeyGen's WebRTC connection.
        """
        logger.info("🔊 Received audio track from HeyGen, starting audio forwarding")
        
        # Forward audio frames from HeyGen to our audio track
        asyncio.create_task(self._forward_audio_frames(track, self._audio_track))
    
    async def _forward_audio_frames(self, source_track: Any, dest_track: Any) -> None:
        """Forward audio frames from HeyGen to agent's audio track.
        
        Args:
            source_track: Audio track from HeyGen.
            dest_track: Agent's audio track to write to.
        """
        try:
            logger.info("🔊 Starting HeyGen audio frame forwarding")
            frame_count = 0
            while True:
                try:
                    # Read audio frame from HeyGen
                    frame = await source_track.recv()
                    frame_count += 1
                    
                    # Convert frame to bytes and write to agent's audio track
                    if hasattr(frame, 'to_ndarray'):
                        audio_array = frame.to_ndarray()
                        
                        # Convert mono to stereo if needed (agent track expects stereo)
                        # HeyGen sends mono (shape=(1, samples)), we need interleaved stereo
                        if audio_array.shape[0] == 1:
                            # Flatten to 1D array of samples
                            mono_samples = audio_array.flatten()
                            
                            # Create stereo by interleaving each mono sample
                            stereo_samples = np.repeat(mono_samples, 2)
                            audio_bytes = stereo_samples.tobytes()
                        else:
                            # Already multi-channel, just flatten and convert
                            audio_bytes = audio_array.flatten().tobytes()
                        
                        # Set flag to allow HeyGen audio through the muted track
                        self._forwarding_audio = True
                        await dest_track.write(audio_bytes)
                        self._forwarding_audio = False
                    else:
                        logger.warning("⚠️ Received frame without to_ndarray() method")
                        
                except Exception as e:
                    if "ended" in str(e).lower() or "closed" in str(e).lower():
                        logger.info(f"🔊 HeyGen audio track ended (forwarded {frame_count} frames)")
                        break
                    else:
                        logger.error(f"❌ Error forwarding audio frame #{frame_count}: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        break
                        
        except Exception as e:
            logger.error(f"❌ Error in audio forwarding loop: {e}")
            import traceback
            logger.error(traceback.format_exc())

    async def _on_text_chunk(self, text_delta: str, item_id: Optional[str]) -> None:
        """Handle text chunk from the LLM.
        
        Accumulates text chunks until a complete sentence or response is ready,
        then sends to HeyGen for lip-sync.
        
        Args:
            text_delta: The text chunk/delta from the LLM.
            item_id: The response item ID.
        """
        # If this is a new response, reset the buffer and sent tracking
        if item_id != self._current_response_id:
            if self._text_buffer:
                # Send any accumulated text from previous response
                await self._send_text_to_heygen(self._text_buffer.strip())
            self._text_buffer = ""
            self._current_response_id = item_id
            self._sent_texts.clear()
        
        # Accumulate text
        self._text_buffer += text_delta
        
        # Send when we have a complete sentence (ending with period, !, or ?)
        # But only if it's substantial enough (> 15 chars) to avoid sending tiny fragments
        # Don't send on commas/semicolons to reduce repetition
        if any(self._text_buffer.rstrip().endswith(p) for p in ['.', '!', '?']):
            text_to_send = self._text_buffer.strip()
            # Only send if it's substantial (>15 chars) and not already sent
            if text_to_send and len(text_to_send) > 15 and text_to_send not in self._sent_texts:
                await self._send_text_to_heygen(text_to_send)
                self._sent_texts.add(text_to_send)
                self._text_buffer = ""  # Clear buffer after sending
            elif text_to_send in self._sent_texts:
                self._text_buffer = ""  # Clear buffer to avoid re-sending
    
    async def _send_text_to_heygen(self, text: str) -> None:
        """Send text to HeyGen for the avatar to speak with lip-sync.
        
        Args:
            text: The text for the avatar to speak.
        """
        if not text:
            return
        
        if not self._connected:
            logger.warning("Cannot send text to HeyGen - not connected")
            return
        
        try:
            logger.info(f"📤 Sending text to HeyGen: '{text[:50]}...'")
            await self.rtc_manager.send_text(text, task_type="repeat")
            logger.debug("✅ Text sent to HeyGen successfully")
        except Exception as e:
            logger.error(f"❌ Failed to send text to HeyGen: {e}")
            import traceback
            logger.error(traceback.format_exc())

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

