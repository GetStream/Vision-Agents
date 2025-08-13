"""
Agent implementation for Stream video call integration.

This module provides the Agent class that allows for easy integration of AI agents
into Stream video calls with support for tools, pre-processors, and various AI services.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Any, Callable, List, Optional, Protocol
from uuid import uuid4

import aiortc
from PIL import Image
from getstream.video import rtc
from getstream.video.rtc import audio_track
from getstream.video.rtc.tracks import (
    SubscriptionConfig,
    TrackSubscriptionConfig,
    TrackType,
)

# Import STT, TTS, and VAD base classes from stream-py package
from getstream.plugins.common.stt import STT
from getstream.plugins.common.tts import TTS
from getstream.plugins.common.vad import VAD


class Tool(Protocol):
    """Protocol for agent tools."""

    def __call__(self, *args, **kwargs) -> Any:
        """Execute the tool."""
        ...


class PreProcessor(Protocol):
    """Protocol for pre-processors."""

    def process(self, data: Any) -> Any:
        """Process input data."""
        ...


class Model(Protocol):
    """Protocol for AI models."""

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from the model."""
        ...


# STT and TTS are now imported directly from stream-py package


class TurnDetection(Protocol):
    """Protocol for turn detection services."""

    def detect_turn(self, audio_data: bytes) -> bool:
        """Detect if it's the agent's turn to speak."""
        ...


class ImageProcessor(Protocol):
    """Protocol for image processors."""

    async def process_image(
        self, image: Image.Image, user_id: str, metadata: dict = None
    ) -> None:
        """Process a video frame image."""
        ...


class Agent:
    """
    AI Agent that can join Stream video calls and interact with participants.

    Example usage:
        agent = Agent(
            instructions="Roast my in-game performance in a funny but encouraging manner",
            pre_processors=[Roboflow(), dota_api("gameid")],
            model=openai_model,
            stt=speech_to_text,
            tts=text_to_speech,
            turn_detection=turn_detector
        )
        await agent.join(call)
    """

    def __init__(
        self,
        instructions: str,
        tools: Optional[List[Tool]] = None,
        pre_processors: Optional[List[PreProcessor]] = None,
        model: Optional[Model] = None,
        stt: Optional[STT] = None,
        tts: Optional[TTS] = None,
        vad: Optional[VAD] = None,
        turn_detection: Optional[TurnDetection] = None,
        image_interval: Optional[int] = None,
        image_processors: Optional[List[ImageProcessor]] = None,
        target_user_id: Optional[str] = None,
        bot_id: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize the Agent.

        Args:
            instructions: System instructions for the agent
            tools: List of tools the agent can use
            pre_processors: List of pre-processors for input data
            model: AI model for generating responses
            stt: Speech-to-Text service
            tts: Text-to-Speech service
            vad: Voice Activity Detection service (optional)
            turn_detection: Turn detection service
            image_interval: Interval in seconds for image processing (None to disable)
            image_processors: List of image processors to apply to video frames
            target_user_id: Specific user to capture video from (None for all users)
            bot_id: Unique bot ID (auto-generated if not provided)
            name: Display name for the bot
        """
        self.instructions = instructions
        self.tools = tools or []
        self.pre_processors = pre_processors or []
        self.model = model
        self.stt = stt
        self.tts = tts
        self.vad = vad
        self.turn_detection = turn_detection
        self.image_interval = image_interval
        self.image_processors = image_processors or []
        self.target_user_id = target_user_id
        self.bot_id = bot_id or f"agent-{uuid4()}"
        self.name = name or "AI Agent"

        self._connection: Optional[rtc.RTCConnection] = None
        self._audio_track: Optional[audio_track.AudioStreamTrack] = None
        self._is_running = False
        self._callback_executed = False

        self.logger = logging.getLogger(f"Agent[{self.bot_id}]")

    async def _process_video_track(self, track_id: str, track_type: str, user):
        """Process video frames from a specific track."""
        self.logger.info(
            f"🎥 Processing video track: {track_id} from user {user.user_id} (type: {track_type})"
        )

        # Only process video tracks
        if track_type != "video":
            self.logger.debug(f"Ignoring non-video track: {track_type}")
            return

        # If target_user_id is specified, only process that user's video
        if self.target_user_id and user.user_id != self.target_user_id:
            self.logger.debug(
                f"Ignoring video from user {user.user_id} (target: {self.target_user_id})"
            )
            return

        # Subscribe to the video track
        track = self._connection.subscriber_pc.add_track_subscriber(track_id)
        if not track:
            self.logger.error(f"❌ Failed to subscribe to track: {track_id}")
            return

        self.logger.info(
            f"✅ Successfully subscribed to video track from {user.user_id}"
        )

        try:
            while True:
                try:
                    # Receive video frame
                    video_frame: aiortc.mediastreams.VideoFrame = await track.recv()
                    if not video_frame:
                        continue

                    # Convert to PIL Image
                    img = video_frame.to_image()

                    # Process through all image processors
                    for processor in self.image_processors:
                        try:
                            await processor.process_image(
                                img,
                                user.user_id,
                                metadata={
                                    "track_id": track_id,
                                    "timestamp": asyncio.get_event_loop().time(),
                                },
                            )
                        except Exception as e:
                            self.logger.error(
                                f"❌ Error in image processor {type(processor).__name__}: {e}"
                            )

                except Exception as e:
                    if "Connection closed" in str(e) or "Track ended" in str(e):
                        self.logger.info(
                            f"🔌 Video track ended for user {user.user_id}"
                        )
                        break
                    else:
                        self.logger.error(f"❌ Error processing video frame: {e}")
                        self.logger.error(traceback.format_exc())
                        await asyncio.sleep(1)  # Brief pause before retry

        except Exception as e:
            self.logger.error(f"❌ Fatal error in video processing: {e}")
            self.logger.error(traceback.format_exc())

    async def join(
        self,
        call,
        user_creation_callback: Optional[Callable] = None,
        on_connected_callback: Optional[Callable] = None,
    ) -> None:
        """
        Join a Stream video call.

        Args:
            call: Stream video call object
            user_creation_callback: Optional callback to create the bot user
            on_connected_callback: Optional async callback that receives (agent, connection)
                                 and runs as a background task after connection is established
        """
        if self._is_running:
            raise RuntimeError("Agent is already running")

        self.logger.info(f"🤖 Agent joining call: {call.id}")

        # Create bot user if callback provided
        if user_creation_callback:
            user_creation_callback(self.bot_id, self.name)

        # Set up audio track if TTS is available
        if self.tts:
            self._audio_track = audio_track.AudioStreamTrack(framerate=16000)
            self.tts.set_output_track(self._audio_track)

        try:
            # Configure subscription based on whether video processing is enabled
            subscription_config = None
            if self.image_processors:
                subscription_config = SubscriptionConfig(
                    default=TrackSubscriptionConfig(
                        track_types=[
                            TrackType.TRACK_TYPE_VIDEO,
                            TrackType.TRACK_TYPE_AUDIO,
                        ]
                    )
                )

            async with await rtc.join(
                call, self.bot_id, subscription_config=subscription_config
            ) as connection:
                self._connection = connection
                self._is_running = True

                self.logger.info(f"🤖 Agent joined call: {call.id}")

                # Set up audio track if available
                if self._audio_track:
                    await connection.add_tracks(audio=self._audio_track)
                    self.logger.info("🤖 Agent ready to speak")

                # Set up event handlers
                await self._setup_event_handlers()

                # Execute callback after full initialization to prevent race conditions
                if on_connected_callback and not self._callback_executed:
                    self._callback_executed = True

                    async def safe_callback():
                        try:
                            await asyncio.wait_for(
                                on_connected_callback(self, self._connection),
                                timeout=30.0,
                            )
                        except asyncio.TimeoutError:
                            self.logger.error(
                                "❌ on_connected_callback timed out after 30 seconds"
                            )
                        except Exception as e:
                            self.logger.error(f"❌ Error in on_connected_callback: {e}")
                            self.logger.error(traceback.format_exc())

                    asyncio.create_task(safe_callback())

                try:
                    self.logger.info("🎧 Agent is active - press Ctrl+C to stop")
                    await connection.wait()
                except Exception as e:
                    self.logger.error(f"❌ Error during agent operation: {e}")
                    self.logger.error(traceback.format_exc())

        except asyncio.CancelledError:
            self.logger.info("Stopping agent...")
        except Exception as e:
            # Handle cleanup errors gracefully
            if "NoneType" in str(e) and "await" in str(e):
                self.logger.warning(
                    "Cleanup error (likely WebSocket already closed) - ignoring"
                )
            else:
                self.logger.error(f"Error during agent operation: {e}")
                self.logger.error(traceback.format_exc())
                raise
        finally:
            self._is_running = False
            self._connection = None

            if self.stt:
                try:
                    await self.stt.close()
                except Exception as e:
                    self.logger.warning(f"Error closing STT service: {e}")
                    self.logger.error(traceback.format_exc())

    async def say(self, message: str) -> None:
        """Send a message via TTS."""
        if not self.tts:
            return

        # Check if connection exists
        if not self._connection:
            self.logger.error(
                "❌ Cannot send message: Agent is not connected to a call"
            )
            return

        max_retries = 3
        retry_delay = 1.0  # Start with 1 second delay

        for attempt in range(max_retries):
            try:
                if self._connection.publisher_pc is not None:
                    self.logger.debug(
                        f"🔊 Waiting for publisher connection (attempt {attempt + 1}/{max_retries})"
                    )

                    # Wait for connection with timeout
                    await asyncio.wait_for(
                        self._connection.publisher_pc.wait_for_connected(), timeout=10.0
                    )

                    # Double-check the connection state
                    if self._connection.publisher_pc.connectionState == "connected":
                        self.logger.info(
                            "🤖 Agent ready to speak - TTS audio track published"
                        )
                        await self.tts.send(message)
                        break
                    else:
                        raise RuntimeError(
                            f"Publisher connection state is {self._connection.publisher_pc.connectionState}, not connected"
                        )

                else:
                    raise RuntimeError("No publisher peer connection available")

            except (asyncio.TimeoutError, RuntimeError) as e:
                if attempt < max_retries - 1:
                    self.logger.warning(
                        f"⚠️ TTS setup attempt {attempt + 1} failed: {e}, retrying in {retry_delay}s"
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    self.logger.error(
                        f"❌ Failed to set up TTS after {max_retries} attempts: {e}"
                    )
                    self.logger.error(traceback.format_exc())
                    # Clean up on failure
                    if self._audio_track:
                        self._audio_track = None
                    return

    async def _setup_event_handlers(self) -> None:
        """Set up event handlers for the connection."""
        if not self._connection:
            self.logger.error("❌ No active connections found")
            return

        # Handle new participants
        async def on_track_published(event):
            try:
                user_id = "unknown"
                if hasattr(event, "participant") and event.participant:
                    user_id = getattr(event.participant, "user_id", "unknown")

                track_id = getattr(event, "track_id", "unknown")
                track_type = getattr(event, "track_type", "unknown")

                self.logger.info(
                    f"Handling track published: {user_id} - {track_id} - {track_type}"
                )

                if user_id and user_id != self.bot_id:
                    self.logger.info(f"👋 New participant joined: {user_id}")
                    await self._handle_new_participant(user_id)
                elif user_id == self.bot_id:
                    self.logger.debug(f'Not subscribing to track: user_id: "{user_id}"')
            except Exception as e:
                self.logger.error(f"❌ Error handling track published event: {e}")
                self.logger.error(traceback.format_exc())

        # Handle audio data for STT using Stream SDK pattern
        @self._connection.on("audio")
        async def on_audio_received(pcm, user):
            """Handle incoming audio data from participants."""
            try:
                if self.stt and user and user != self.bot_id:
                    await self._handle_audio_input(pcm, user)
            except Exception as e:
                self.logger.error(f"Error handling audio received event: {e}")
                self.logger.error(traceback.format_exc())

        # Set up video track handler if image processors are configured
        if self.image_processors and self._connection:

            def on_track_added(track_id, track_type, user):
                user_id = user.user_id if user else "unknown"
                self.logger.info(
                    f"🎬 New track detected: {track_id} ({track_type}) from {user_id}"
                )
                if track_type == "video":
                    asyncio.create_task(
                        self._process_video_track(track_id, track_type, user)
                    )

            self._connection.on("track_added", on_track_added)

        if self.tts:

            def on_tts_audio(audio_data, user=None, metadata=None):
                user_id = user.user_id if user else "unknown"
                self.logger.debug(f"🔊 TTS audio generated for: {user_id}")
                asyncio.create_task(self._on_tts_audio(audio_data, user, metadata))

            self.tts.on("audio", on_tts_audio)

            def on_tts_error(error, user=None, metadata=None):
                user_id = user.user_id if user else "unknown"
                self.logger.error(f"❌ TTS Error for {user_id}: {error}")
                asyncio.create_task(self._on_tts_error(error, user, metadata))

            self.tts.on("error", on_tts_error)

        # Set up WebSocket event handlers to keep connection alive
        try:
            if hasattr(self._connection, "_ws_client") and self._connection._ws_client:
                # Listen for track events to keep the connection active
                self._connection._ws_client.on_event(
                    "track_published", on_track_published
                )
                self._connection._ws_client.on_event(
                    "track_unpublished",
                    lambda event: self.logger.debug("Track unpublished"),
                )

        except Exception as e:
            self.logger.error(f"Error setting up WebSocket event handlers: {e}")
            self.logger.error(traceback.format_exc())

    async def _handle_audio_input(self, pcm_data, user) -> None:
        """Handle incoming audio data from Stream WebRTC connection."""
        if not self.stt:
            return

        try:
            # Check if it's our turn to respond (if turn detection is configured)
            if self.turn_detection and hasattr(pcm_data, "data"):
                if not self.turn_detection.detect_turn(pcm_data.data):
                    return

            # Set up event listeners for transcription results (one-time setup)
            if not hasattr(self, "_stt_setup"):
                self.stt.on("transcript", self._on_transcript)
                self.stt.on("partial_transcript", self._on_partial_transcript)
                self.stt.on("error", self._on_stt_error)
                self._stt_setup = True

            # Handle audio processing with or without VAD
            if self.vad:
                # With VAD: Only process audio when speech is detected
                await self._process_audio_with_vad(pcm_data, user)
            else:
                # Without VAD: Process all audio directly through STT
                await self.stt.process_audio(pcm_data, user)

        except Exception as e:
            self.logger.error(f"Error handling audio input from user {user}: {e}")
            self.logger.error(traceback.format_exc())

    async def _process_audio_with_vad(self, pcm_data, user) -> None:
        """Process audio with Voice Activity Detection."""
        try:
            # Set up VAD event listeners (one-time setup)
            if not hasattr(self, "_vad_setup"):
                self.vad.on("speech_start", self._on_speech_start)
                self.vad.on("speech_end", self._on_speech_end)
                self._vad_setup = True

            # Process audio through VAD first
            await self.vad.process_audio(pcm_data, user)

            # VAD will trigger speech events that route to STT when appropriate

        except Exception as e:
            self.logger.error(f"Error processing audio with VAD for user {user}: {e}")
            self.logger.error(traceback.format_exc())

    async def _on_speech_start(self, user=None, metadata=None):
        """Handle start of speech detected by VAD."""
        user_info = (
            user.name
            if user and hasattr(user, "name")
            else (user.user_id if user and hasattr(user, "user_id") else "unknown")
        )
        self.logger.debug(f"🎙️ Speech started: {user_info}")

    async def _on_speech_end(self, user=None, metadata=None):
        """Handle end of speech detected by VAD."""
        user_info = (
            user.name
            if user and hasattr(user, "name")
            else (user.user_id if user and hasattr(user, "user_id") else "unknown")
        )
        self.logger.debug(f"🎙️ Speech ended: {user_info}")

    async def _on_transcript(self, text: str, user=None, metadata=None):
        """Handle final transcript from STT service."""
        if text and text.strip():
            user_info = (
                user.name
                if user and hasattr(user, "name")
                else (user.user_id if user and hasattr(user, "user_id") else "unknown")
            )
            self.logger.info(f"🎤 [{user_info}]: {text}")

            # Log confidence if available
            if metadata and metadata.get("confidence"):
                self.logger.debug(f"    └─ confidence: {metadata['confidence']:.2%}")

            await self._process_transcription(text, user)

    async def _on_partial_transcript(self, text: str, user=None, metadata=None):
        """Handle partial transcript from STT service."""
        if text and text.strip():
            user_info = (
                user.name
                if user and hasattr(user, "name")
                else (user.user_id if user and hasattr(user, "user_id") else "unknown")
            )
            self.logger.debug(f"🎤 [{user_info}] (partial): {text}")

    async def _on_stt_error(self, error):
        """Handle STT service errors."""
        self.logger.error(f"❌ STT Error: {error}")

    async def _on_tts_audio(self, audio_data, user=None, metadata=None):
        """Handle TTS audio generation events."""
        try:
            user_info = (
                user.name
                if user and hasattr(user, "name")
                else (user.user_id if user and hasattr(user, "user_id") else "agent")
            )
            self.logger.debug(f"🔊 TTS audio generated for: {user_info}")

            # TTS service automatically handles audio output to the configured track

        except Exception as e:
            self.logger.error(f"Error handling TTS audio event: {e}")
            self.logger.error(traceback.format_exc())

    async def _on_tts_error(self, error, user=None, metadata=None):
        """Handle TTS service errors."""
        try:
            user_info = (
                user.name
                if user and hasattr(user, "name")
                else (user.user_id if user and hasattr(user, "user_id") else "agent")
            )
            self.logger.error(f"❌ TTS Error for {user_info}: {error}")
        except Exception as e:
            self.logger.error(f"Error handling TTS error event: {e}")
            self.logger.error(traceback.format_exc())

    async def _process_transcription(self, text: str, user=None) -> None:
        """Process a complete transcription and generate response."""
        try:
            # Process with pre-processors
            processed_data = text
            for processor in self.pre_processors:
                processed_data = processor.process(processed_data)

            # Generate response using model
            if self.model:
                response = await self._generate_response(processed_data)

                # Send response via TTS
                if self.tts and response:
                    await self.tts.send(response)
                    self.logger.info(f"🤖 Responded: {response}")

        except Exception as e:
            self.logger.error(f"Error processing transcription: {e}")
            self.logger.error(traceback.format_exc())

    async def _generate_response(self, input_text: str) -> str:
        """Generate a response using the AI model."""
        if not self.model:
            return ""

        try:
            # Create context with instructions and available tools
            context = f"""
            System: {self.instructions}
            
            Available tools: {[str(tool) for tool in self.tools]}
            
            User input: {input_text}
            
            Respond appropriately based on your instructions.
            """

            response = await self.model.generate(context)
            return response

        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            self.logger.error(traceback.format_exc())
            return "I'm sorry, I encountered an error processing your request."

    def is_running(self) -> bool:
        """Check if the agent is currently running."""
        return self._is_running

    async def _generate_greeting(self, participant_count: int) -> str:
        """Generate a greeting message when joining a call."""
        if self.model:
            try:
                context = f"""
                System: {self.instructions}
                
                You are joining a video call with {participant_count} participant(s).
                Generate a brief, friendly greeting to introduce yourself.
                Keep it under 2 sentences.
                """

                response = await self.model.generate(context)
                return response
            except Exception as e:
                self.logger.error(f"Error generating greeting: {e}")
                return f"Hello everyone! I'm {self.name}"
        else:
            return f"Hello everyone! I'm {self.name}"

    async def _generate_participant_greeting(self, user_id: str) -> str:
        """Generate a greeting message for a new participant."""
        if self.model:
            try:
                context = f"""
                System: {self.instructions}
                
                A new participant (user-{user_id}) has joined the call.
                Generate a brief, friendly greeting to welcome them.
                Keep it under 2 sentences.
                """

                response = await self.model.generate(context)
                return response
            except Exception as e:
                self.logger.error(f"Error generating participant greeting: {e}")
                return f"Welcome to the call, user-{user_id}!"
        else:
            return f"Welcome {user_id}!"

    async def _handle_new_participant(self, user_id: str) -> None:
        """Handle when a new participant joins the call."""
        try:
            if self.tts:
                greeting = await self._generate_participant_greeting(user_id)
                if greeting:
                    await self.tts.send(greeting)
                    self.logger.info(
                        f"🤖 Welcomed new participant {user_id}: {greeting}"
                    )
        except Exception as e:
            self.logger.error(f"Error handling new participant {user_id}: {e}")
            self.logger.error(traceback.format_exc())

    async def stop(self) -> None:
        """Stop the agent and clean up resources."""
        try:
            self._is_running = False

            # Close STT service if available
            if self.stt and hasattr(self.stt, "close"):
                try:
                    await self.stt.close()
                except Exception as e:
                    self.logger.error(f"Error closing STT service: {e}")

            # Close VAD service if available
            if self.vad and hasattr(self.vad, "close"):
                try:
                    await self.vad.close()
                except Exception as e:
                    self.logger.error(f"Error closing VAD service: {e}")

            # Close TTS service if available
            if self.tts and hasattr(self.tts, "close"):
                try:
                    await self.tts.close()
                except Exception as e:
                    self.logger.error(f"Error closing TTS service: {e}")

            self.logger.info("🛑 Agent stopped and resources cleaned up")

        except Exception as e:
            self.logger.error(f"Error during agent cleanup: {e}")
            self.logger.error(traceback.format_exc())
