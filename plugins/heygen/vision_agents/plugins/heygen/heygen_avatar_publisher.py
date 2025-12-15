import asyncio
import logging
from typing import Any, Optional, Tuple

from getstream.video.rtc import audio_track
from getstream.video.rtc.track_util import PcmData
from vision_agents.core.processors.base_processor import AudioPublisher, VideoPublisher

from .heygen_rtc_manager import HeyGenRTCManager
from .heygen_types import VideoQuality
from .heygen_video_track import HeyGenVideoTrack

logger = logging.getLogger(__name__)


class AvatarPublisher(AudioPublisher, VideoPublisher):
    """HeyGen avatar video and audio publisher.

    Publishes video of a HeyGen avatar that lip-syncs based on LLM text output.

    For standard LLMs: HeyGen provides both video and audio (with TTS).
    For Realtime LLMs: HeyGen provides video only; LLM provides audio.

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
                    quality=heygen.VideoQuality.HIGH
                )
            ]
        )
    """

    name = "heygen_avatar"

    def __init__(
        self,
        avatar_id: str = "default",
        quality: VideoQuality = VideoQuality.HIGH,
        resolution: Tuple[int, int] = (1920, 1080),
        api_key: Optional[str] = None,
        interval: int = 0,
        **kwargs,
    ):
        """Initialize the HeyGen avatar publisher.

        Args:
            avatar_id: HeyGen avatar ID to use for streaming.
            quality: Video quality (VideoQuality.LOW, VideoQuality.MEDIUM, or VideoQuality.HIGH).
            resolution: Output video resolution (width, height).
            api_key: HeyGen API key. Uses HEYGEN_API_KEY env var if not provided.
            interval: Processing interval (not used, kept for compatibility).
            **kwargs: Additional arguments passed to parent class.
        """
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

        # Audio track for publishing HeyGen's audio
        # Create it immediately so the agent can detect it during initialization
        self._audio_track = audio_track.AudioStreamTrack(
            sample_rate=48000, channels=2, format="s16"
        )

        # Connection state
        self._connected = False
        self._connection_task: Optional[asyncio.Task] = None
        self._agent = None  # Will be set by the agent

        # Text buffer for accumulating LLM response chunks before sending to HeyGen
        self._text_buffer = ""
        self._current_response_id: Optional[str] = None
        self._all_sent_texts: set = set()  # Track all sent texts to prevent duplicates

        logger.info(
            f"HeyGen AvatarPublisher initialized "
            f"(avatar: {avatar_id}, quality: {quality}, resolution: {resolution})"
        )

    def publish_audio_track(self):
        """Return the audio track for publishing HeyGen's audio.

        This method is called by the Agent to get the audio track that will
        be published to the call. HeyGen's audio will be forwarded to this track.
        """
        return self._audio_track

    def attach_agent(self, agent: Any) -> None:
        """Attach the agent reference for event subscription.

        This is called automatically by the Agent during initialization.

        Args:
            agent: The agent instance.
        """
        self._agent = agent
        logger.info("Agent reference set for HeyGen avatar publisher")

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
            logger.info("Connected to HeyGen, avatar streaming active")

        except Exception as e:
            logger.error(f"Failed to connect to HeyGen: {e}")
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
            if hasattr(self, "_agent") and self._agent and hasattr(self._agent, "llm"):

                @self._agent.llm.events.subscribe
                async def on_text_chunk(event: LLMResponseChunkEvent):
                    """Handle streaming text chunks from the LLM."""
                    logger.debug(f"HeyGen received text chunk: delta='{event.delta}'")
                    if event.delta:
                        await self._on_text_chunk(event.delta, event.item_id)

                @self._agent.llm.events.subscribe
                async def on_text_complete(event: LLMResponseCompletedEvent):
                    """Handle end of LLM response - split into sentences and send each once."""
                    if not self._text_buffer.strip():
                        return

                    # Split the complete response into sentences
                    import re

                    text = self._text_buffer.strip()
                    # Split on sentence boundaries but keep the punctuation
                    sentences = re.split(r"([.!?]+\s*)", text)
                    # Recombine sentences with their punctuation
                    full_sentences = []
                    for i in range(0, len(sentences) - 1, 2):
                        if sentences[i].strip():
                            sentence = (
                                sentences[i] + sentences[i + 1]
                                if i + 1 < len(sentences)
                                else sentences[i]
                            ).strip()
                            full_sentences.append(sentence)
                    # Handle last part if no punctuation
                    if (
                        sentences
                        and sentences[-1].strip()
                        and not any(
                            sentences[-1].strip().endswith(p) for p in [".", "!", "?"]
                        )
                    ):
                        full_sentences.append(sentences[-1].strip())

                    # Send each sentence once if not already sent
                    for sentence in full_sentences:
                        if sentence and len(sentence) > 5:
                            if sentence not in self._all_sent_texts:
                                await self._send_text_to_heygen(sentence)
                                self._all_sent_texts.add(sentence)
                            else:
                                logger.debug(
                                    f"Skipping duplicate: '{sentence[:30]}...'"
                                )

                    # Reset for next response
                    self._text_buffer = ""
                    self._current_response_id = None

                @self._agent.llm.events.subscribe
                async def on_agent_speech(event: RealtimeAgentSpeechTranscriptionEvent):
                    """Handle agent speech transcription from Realtime LLMs.

                    This is the primary path for Gemini Realtime which transcribes
                    the agent's speech output as text.
                    """
                    logger.debug(f"HeyGen received agent speech: text='{event.text}'")
                    if event.text:
                        # Send directly to HeyGen - this is the complete utterance
                        await self._send_text_to_heygen(event.text)

                logger.info("Subscribed to LLM text output events for HeyGen lip-sync")
            else:
                logger.warning(
                    "Cannot subscribe to text events - no agent or LLM attached yet"
                )
        except Exception as e:
            logger.error(f"Failed to subscribe to text events: {e}", exc_info=True)

    async def _on_video_track(self, track: Any) -> None:
        """Callback when video track is received from HeyGen.

        Args:
            track: Incoming video track from HeyGen's WebRTC connection.
        """
        logger.info("Received video track from HeyGen, starting frame forwarding")
        await self._video_track.start_receiving(track)

    async def _on_audio_track(self, track: Any) -> None:
        """Callback when audio track is received from HeyGen.

        HeyGen provides audio with lip-synced TTS. We forward this audio
        to the agent's audio track so it gets published to the call.

        For Realtime LLMs: We DON'T forward HeyGen audio - the LLM generates its own audio.
        HeyGen is only used for video lip-sync based on text transcriptions.

        Args:
            track: Incoming audio track from HeyGen's WebRTC connection.
        """
        logger.info("Received audio track from HeyGen")

        # Check if we're using a Realtime LLM
        if self._agent and hasattr(self._agent, "llm"):
            from vision_agents.core.llm.realtime import Realtime

            if isinstance(self._agent.llm, Realtime):
                # For Realtime LLMs, don't forward HeyGen audio - use the LLM's native audio
                # HeyGen is only used for lip-synced video based on text transcriptions
                logger.info(
                    "Using Realtime LLM - skipping HeyGen audio forwarding (using LLM's native audio)"
                )
                return

        # For standard LLMs, forward HeyGen's audio to our audio track
        logger.info("Forwarding HeyGen audio to audio track")
        asyncio.create_task(self._forward_audio_frames(track, self._audio_track))

    async def _forward_audio_frames(self, source_track: Any, dest_track: Any) -> None:
        """Forward audio frames from HeyGen to agent's audio track.

        Args:
            source_track: Audio track from HeyGen.
            dest_track: Agent's audio track to write to.
        """
        try:
            logger.info("Starting HeyGen audio frame forwarding")
            frame_count = 0
            while True:
                try:
                    # Read audio frame from HeyGen
                    frame = await source_track.recv()
                    frame_count += 1

                    # Convert AV frame to PcmData
                    pcm = PcmData.from_av_frame(frame)
                    # Resample to match destination track format (48000 Hz, 2 channels)
                    pcm = pcm.resample(target_sample_rate=48000, target_channels=2)
                    await dest_track.write(pcm)

                except Exception as e:
                    if "ended" in str(e).lower() or "closed" in str(e).lower():
                        logger.info(
                            f"HeyGen audio track ended (forwarded {frame_count} frames)"
                        )
                        break
                    logger.error(f"Error forwarding audio frame: {e}", exc_info=True)
                    break

        except Exception as e:
            logger.error(f"Error in audio forwarding loop: {e}", exc_info=True)

    async def _on_text_chunk(self, text_delta: str, item_id: Optional[str]) -> None:
        """Handle text chunk from the LLM.

        Accumulates text chunks. Does NOT send immediately - waits for completion event
        to avoid sending partial/duplicate sentences.

        Args:
            text_delta: The text chunk/delta from the LLM.
            item_id: The response item ID.
        """
        # If this is a new response, reset the buffer and sent tracking
        if item_id != self._current_response_id:
            if self._text_buffer:
                # Send any accumulated text from previous response
                text_to_send = self._text_buffer.strip()
                if text_to_send and text_to_send not in self._all_sent_texts:
                    await self._send_text_to_heygen(text_to_send)
                    self._all_sent_texts.add(text_to_send)
            self._text_buffer = ""
            self._current_response_id = item_id

        # Just accumulate text - don't send yet!
        # Wait for completion event to avoid sending partial sentences
        self._text_buffer += text_delta

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
            logger.info(f"Sending text to HeyGen: '{text[:50]}...'")
            await self.rtc_manager.send_text(text, task_type="repeat")
        except Exception as e:
            logger.error(f"Failed to send text to HeyGen: {e}", exc_info=True)

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

        logger.info("Publishing HeyGen avatar video track")
        return self._video_track

    async def close(self) -> None:
        """Clean up resources and close connections."""
        logger.info("Closing HeyGen avatar publisher")

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
        logger.info("HeyGen avatar publisher closed")
