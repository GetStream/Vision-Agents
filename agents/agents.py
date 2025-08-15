"""
Agent implementation for Stream video call integration.

This module provides the Agent class that allows for easy integration of AI agents
into Stream video calls with support for tools, pre-processors, and various AI services.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Any, Callable, Dict, List, Optional, Protocol
from uuid import uuid4

import aiortc
import av
from PIL import Image
from getstream.video import rtc
from getstream.video.rtc import audio_track
from getstream.video.rtc.tracks import (
    SubscriptionConfig,
    TrackSubscriptionConfig,
    TrackType,
)

# Import STT, TTS, VAD, and STS base classes from stream-py package
from getstream.plugins.common.stt import STT
from getstream.plugins.common.tts import TTS
from getstream.plugins.common.vad import VAD
from getstream.plugins.common.sts import STS
from turn_detection.turn_detection import TurnDetection, TurnEvent


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


class LLM(Protocol):
    """Protocol for AI models."""

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from the model."""
        ...


# STT and TTS are now imported directly from stream-py package


class ImageProcessor(Protocol):
    """Protocol for image processors."""

    async def process_image(
        self, image: Image.Image, user_id: str, metadata: dict = None
    ) -> None:
        """Process a video frame image."""
        ...


class VideoTransformer(Protocol):
    """Protocol for video transformers that modify video frames."""

    async def transform_frame(self, frame: Image.Image) -> Image.Image:
        """Transform a video frame and return the modified frame."""
        ...


class TransformedVideoTrack(aiortc.VideoStreamTrack):
    """Custom video stream track for publishing transformed video frames."""

    def __init__(self):
        super().__init__()
        self.frame_q = asyncio.Queue(
            maxsize=10
        )  # Limit queue size to prevent memory issues
        # Create a subtle blue-tinted frame as fallback (to show the transformer is ready)
        default_frame = Image.new(
            "RGB", (640, 480), color=(20, 30, 80)
        )  # Subtle blue-gray
        self.last_frame = default_frame

    async def add_frame(self, pil_image: Image.Image):
        """Add a transformed PIL Image frame to be published."""
        try:
            # Add frame to queue (drop old frames if queue is full)
            try:
                self.frame_q.put_nowait(pil_image)
            except asyncio.QueueFull:
                # Drop the oldest frame and add the new one
                try:
                    self.frame_q.get_nowait()
                    self.frame_q.put_nowait(pil_image)
                except asyncio.QueueEmpty:
                    pass

        except Exception as e:
            print(f"Error adding frame to video track: {e}")

    async def recv(self) -> av.VideoFrame:
        """Return the next video frame for WebRTC transmission."""
        try:
            # Try to get the latest frame from queue with a very short timeout
            frame = await asyncio.wait_for(self.frame_q.get(), timeout=0.02)
            if frame:
                self.last_frame = frame
        except asyncio.TimeoutError:
            # Use the last frame if no new frame is available - this is critical!
            pass
        except Exception as e:
            print(f"Error getting frame from queue: {e}")

        # Always return a frame - this is essential for continuous video
        try:
            # Get proper timestamp using aiortc's timing
            pts, time_base = await self.next_timestamp()

            # Convert PIL Image to av.VideoFrame
            av_frame = av.VideoFrame.from_image(self.last_frame)
            av_frame.pts = pts
            av_frame.time_base = time_base

            return av_frame
        except Exception as e:
            print(f"Error creating av.VideoFrame: {e}")
            # Fallback: create a simple frame
            pts, time_base = await self.next_timestamp()
            fallback_frame = Image.new("RGB", (640, 480), color=(100, 100, 100))  # Gray
            av_frame = av.VideoFrame.from_image(fallback_frame)
            av_frame.pts = pts
            av_frame.time_base = time_base
            return av_frame


class Agent:
    """
    AI Agent that can join Stream video calls and interact with participants.

    Note that the agent can run in several different modes:
    - STS Model (Speech-to-Speech with OpenAI Realtime API)
    - STT -> Model -> TTS (Traditional pipeline)
    - Video AI/coach
    - Video transformation

    With either a mix or match of those.

    Example usage:
        # Traditional STT -> Model -> TTS pipeline
        agent = Agent(
            pre_processors=[Roboflow(), dota_api("gameid")],
            model=openai_model,
            stt=speech_to_text,
            tts=text_to_speech,
            turn_detection=turn_detector
        )

        # OpenAI Realtime STS mode
        agent = Agent(
            sts_model=openai_realtime_sts
        )

        await agent.join(call)
    """

    def __init__(
        self,
        tools: Optional[List[Tool]] = None,
        pre_processors: Optional[List[PreProcessor]] = None,
        llm: Optional[LLM] = None,
        stt: Optional[STT] = None,
        tts: Optional[TTS] = None,
        vad: Optional[VAD] = None,
        turn_detection: Optional[TurnDetection] = None,
        sts_model: Optional[STS] = None,
        image_interval: Optional[int] = None,
        image_processors: Optional[List[ImageProcessor]] = None,
        video_transformer: Optional[VideoTransformer] = None,
        target_user_id: Optional[str] = None,
        bot_id: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize the Agent.

        Args:
            tools: List of tools the agent can use
            pre_processors: List of pre-processors for input data
            llm: AI model for generating responses
            stt: Speech-to-Text service
            tts: Text-to-Speech service
            vad: Voice Activity Detection service (optional)
            turn_detection: Turn detection service
            sts_model: Speech-to-Speech model (OpenAI Realtime API)
                      When provided, stt and tts are ignored
            image_interval: Interval in seconds for image processing (None to disable)
            image_processors: List of image processors to apply to video frames
            video_transformer: Video transformer to modify video frames before processing
            target_user_id: Specific user to capture video from (None for all users)
            bot_id: Unique bot ID (auto-generated if not provided)
            name: Display name for the bot
        """
        self.tools = tools or []
        self.pre_processors = pre_processors or []
        self.llm = llm
        self.stt = stt
        self.tts = tts
        self.vad = vad
        self.turn_detection = turn_detection
        self.sts_model = sts_model
        self.image_interval = image_interval
        self.image_processors = image_processors or []
        self.video_transformer = video_transformer
        self.target_user_id = target_user_id
        self.bot_id = bot_id or f"agent-{uuid4()}"
        self.name = name or "AI Agent"

        # For STS + interval processing
        self._current_frame = None
        self._interval_task = None

        self._connection: Optional[rtc.RTCConnection] = None
        self._audio_track: Optional[audio_track.AudioStreamTrack] = None
        self._video_track: Optional[TransformedVideoTrack] = None
        self._is_running = False
        self._callback_executed = False

        self.logger = logging.getLogger(f"Agent[{self.bot_id}]")

        # Validate STS vs STT/TTS configuration
        if sts_model and (stt or tts):
            raise ValueError(
                "Cannot use both sts_model and stt/tts. "
                "STS (Speech-to-Speech) models handle both speech-to-text and text-to-speech internally."
            )

        if sts_model and llm:
            self.logger.warning(
                "Using STS model with a separate llm parameter. "
                "The STS model will handle conversation flow, and the llm parameter will be ignored."
            )

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

                    # Apply video transformation if configured
                    if self.video_transformer:
                        try:
                            img = await self.video_transformer.transform_frame(img)

                            # Publish transformed frame to video track
                            if self._video_track:
                                await self._video_track.add_frame(img)

                        except Exception as e:
                            self.logger.error(
                                f"❌ Error in video transformer {type(self.video_transformer).__name__}: {e}"
                            )

                    # Store current frame for interval processing
                    self._current_frame = img

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

        # Handle STS model connection separately
        if self.sts_model:
            try:
                self.logger.info("🤖 Connecting STS model to call")

                # For STS mode, we need both STS connection and video processing if interval is specified
                if self.image_interval and (
                    self.pre_processors or self.image_processors
                ):
                    # STS with interval processing - need WebRTC for video
                    await self._handle_sts_with_interval_processing(
                        call, user_creation_callback, on_connected_callback
                    )
                else:
                    # Pure STS mode - no video processing needed
                    await self._handle_pure_sts_mode(call, on_connected_callback)

                return

            except Exception as e:
                self.logger.error(f"❌ Failed to connect STS model: {e}")
                raise

        # Set up audio track if TTS is available (traditional mode)
        if self.tts:
            self._audio_track = audio_track.AudioStreamTrack(framerate=16000)
            self.tts.set_output_track(self._audio_track)

        # Set up video track if video transformer is available
        if self.video_transformer:
            self._video_track = TransformedVideoTrack()
            self.logger.info("🎥 Video track initialized for transformation publishing")

        try:
            # Configure subscription based on what features are enabled
            subscription_config = None
            track_types = []

            # Subscribe to audio if we have STT or turn detection
            if self.stt or self.turn_detection:
                track_types.append(TrackType.TRACK_TYPE_AUDIO)

            # Subscribe to video if we have image processors or video transformer
            if self.image_processors or self.video_transformer:
                track_types.append(TrackType.TRACK_TYPE_VIDEO)

            # Create subscription config if we need any tracks
            if track_types:
                subscription_config = SubscriptionConfig(
                    default=TrackSubscriptionConfig(track_types=track_types)
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

                # Set up video track if available
                if self._video_track:
                    await connection.add_tracks(video=self._video_track)
                    self.logger.info("🎥 Agent ready to publish transformed video")

                # Set up event handlers
                await self._setup_event_handlers()

                # Set up turn detection callbacks (start happens later)
                if self.turn_detection:
                    self._setup_turn_detection_callbacks()

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

                # Start turn detection last, after event handlers and track subscriptions are ready
                if self.turn_detection:
                    try:
                        self.turn_detection.start()
                        self.logger.info("Turn detection started")
                    except Exception as e:
                        self.logger.error(
                            f"Failed to start turn detection: {e}", exc_info=True
                        )

                try:
                    self.logger.info("🎧 Agent is active - press Ctrl+C to stop")
                    self.logger.debug("Waiting for connection to end...")
                    await connection.wait()
                    self.logger.info("Connection ended normally")
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

            # Stop turn detection if available
            if self.turn_detection:
                try:
                    self.turn_detection.stop()
                    self.logger.info("Turn detection stopped")
                except Exception as e:
                    self.logger.warning(
                        f"Error stopping turn detection: {e}", exc_info=True
                    )

            if self.stt:
                try:
                    await self.stt.close()
                except Exception as e:
                    self.logger.warning(f"Error closing STT service: {e}")
                    self.logger.error(traceback.format_exc())

    async def _handle_pure_sts_mode(self, call, on_connected_callback):
        """Handle pure STS mode without video processing."""
        async with await self.sts_model.connect(
            call, agent_user_id=self.bot_id
        ) as sts_connection:
            self.logger.info("✅ STS model connected successfully")
            self._sts_connection = sts_connection
            self._is_running = True

            # Execute callback after initialization
            if on_connected_callback and not self._callback_executed:
                self._callback_executed = True
                asyncio.create_task(self._safe_sts_callback(on_connected_callback))

            # Send initial greeting to activate the STS agent
            if hasattr(self.sts_model, "send_user_message"):
                await self.sts_model.send_user_message(
                    "Please give a brief, friendly greeting to welcome the user to the call."
                )
            elif hasattr(self.sts_model, "send_message"):
                await self.sts_model.send_message(
                    "Please give a brief, friendly greeting to welcome the user to the call."
                )

            self.logger.info("🎧 STS Agent is active - press Ctrl+C to stop")

            # Process STS events - this keeps the connection alive and handles audio
            async for event in sts_connection:
                self.logger.debug(f"🔔 STS Event: {event.type}")
                # Handle any STS-specific events here if needed

    async def _handle_sts_with_interval_processing(
        self, call, user_creation_callback, on_connected_callback
    ):
        """Handle STS mode with interval-based video processing."""
        # First establish STS connection
        async with await self.sts_model.connect(
            call, agent_user_id=self.bot_id
        ) as sts_connection:
            self.logger.info("✅ STS model connected successfully")
            self._sts_connection = sts_connection

            # Also join WebRTC for video processing
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
            ) as rtc_connection:
                self._connection = rtc_connection
                self._is_running = True

                self.logger.info("✅ Agent joined for video processing")

                # Set up video processing
                await self._setup_event_handlers()

                # Execute callback after initialization
                if on_connected_callback and not self._callback_executed:
                    self._callback_executed = True
                    asyncio.create_task(self._safe_sts_callback(on_connected_callback))

                # Start interval processing task
                if self.image_interval:
                    self._interval_task = asyncio.create_task(
                        self._interval_processing_loop()
                    )

                # Send initial greeting
                if hasattr(self.sts_model, "send_message"):
                    await self.sts_model.send_message(
                        f"Hello! I'm your Dota 2 coach. I'll be analyzing your gameplay every {self.image_interval} seconds and providing real-time feedback. Let's dominate this match!"
                    )

                self.logger.info(
                    f"🎧 STS Agent with interval processing active (every {self.image_interval}s) - press Ctrl+C to stop"
                )

                # Process STS events in background
                async def process_sts_events():
                    try:
                        async for event in sts_connection:
                            self.logger.debug(f"🔔 STS Event: {event.type}")
                    except Exception as e:
                        self.logger.error(f"❌ Error processing STS events: {e}")

                sts_task = asyncio.create_task(process_sts_events())

                try:
                    # Wait for WebRTC connection to end
                    await rtc_connection.wait()
                finally:
                    # Clean up tasks
                    if self._interval_task:
                        self._interval_task.cancel()
                        try:
                            await self._interval_task
                        except asyncio.CancelledError:
                            pass

                    sts_task.cancel()
                    try:
                        await sts_task
                    except asyncio.CancelledError:
                        pass

    async def _interval_processing_loop(self):
        """Run pre-processors and send data to STS model at regular intervals."""
        while self._is_running:
            try:
                await asyncio.sleep(self.image_interval)

                if not self._is_running:
                    break

                # Get current frame
                current_frame = self._current_frame
                if current_frame is None:
                    self.logger.debug("No current frame available for processing")
                    continue

                self.logger.debug(
                    f"🔄 Running interval processing (frame: {current_frame.size if current_frame else None})"
                )

                # Process through pre-processors
                processed_data = {}
                for i, processor in enumerate(self.pre_processors):
                    try:
                        # Check if processor has async process method
                        if hasattr(
                            processor, "process"
                        ) and asyncio.iscoroutinefunction(processor.process):
                            result = await processor.process(current_frame)
                        else:
                            result = processor.process(current_frame)
                        processed_data[f"processor_{i}_{type(processor).__name__}"] = (
                            result
                        )
                        self.logger.debug(
                            f"✅ Processed through {type(processor).__name__}"
                        )
                    except Exception as e:
                        self.logger.error(
                            f"❌ Error in processor {type(processor).__name__}: {e}"
                        )
                        processed_data[
                            f"processor_{i}_{type(processor).__name__}_error"
                        ] = str(e)

                # Send multimodal data to STS model
                if hasattr(self.sts_model, "send_multimodal_data"):
                    context_text = self._format_coaching_context(processed_data)
                    await self.sts_model.send_multimodal_data(
                        text=context_text, image=current_frame, data=processed_data
                    )
                    self.logger.debug("📤 Sent multimodal data to STS model")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Error in interval processing: {e}")
                self.logger.error(traceback.format_exc())

    def _format_coaching_context(self, processed_data: Dict) -> str:
        """Format processed data into coaching context for the STS model."""
        context_parts = []

        for processor_name, data in processed_data.items():
            if "error" in processor_name:
                continue

            if "YOLOProcessor" in processor_name and isinstance(data, dict):
                # Format YOLO data
                analysis = data.get("analysis", {})
                if analysis.get("team_fight_detected"):
                    context_parts.append(
                        "🔥 TEAM FIGHT DETECTED! Analyze positioning and decision making."
                    )
                if analysis.get("farming_opportunity"):
                    context_parts.append("🌾 Good farming opportunity available.")
                if analysis.get("danger_level") == "high":
                    context_parts.append(
                        "⚠️ HIGH DANGER - Player may be in risky position."
                    )

            elif "DotaAPI" in processor_name and isinstance(data, dict):
                # Format Dota API data
                analysis = data.get("analysis", {})

                if analysis.get("issues"):
                    issues = ", ".join(analysis["issues"])
                    context_parts.append(f"Issues detected: {issues}")

                if analysis.get("performance_score"):
                    score = analysis["performance_score"]
                    context_parts.append(f"Performance score: {score:.1f}/100")

                if data.get("recommendations"):
                    context_parts.append("Recommendations available in data.")

        if context_parts:
            return f"Current game analysis: {' '.join(context_parts)}"
        else:
            return "Analyzing current gameplay state..."

    async def _safe_sts_callback(self, on_connected_callback):
        """Safely execute the on_connected_callback for STS mode."""
        try:
            await asyncio.wait_for(
                on_connected_callback(self, self._sts_connection),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            self.logger.error("❌ on_connected_callback timed out after 30 seconds")
        except Exception as e:
            self.logger.error(f"❌ Error in on_connected_callback: {e}")
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
                    self.logger.debug(f"Skipping bot's own track: {user_id}")
            except Exception as e:
                self.logger.error(f"❌ Error handling track published event: {e}")
                self.logger.error(traceback.format_exc())

        # Handle audio data for STT and turn detection using Stream SDK pattern
        @self._connection.on("audio")
        async def on_audio_received(pcm, user):
            """Handle incoming audio data from participants."""
            try:
                # Skip if it's the bot's own audio
                if user is not None and (
                    (hasattr(user, "user_id") and user.user_id == self.bot_id)
                    or (isinstance(user, str) and user == self.bot_id)
                ):
                    return

                # Log audio arrival for diagnostics
                try:
                    length = len(pcm.data) if hasattr(pcm, "data") else len(pcm)
                except Exception:
                    length = -1
                uid = user.user_id if hasattr(user, "user_id") else str(user)
                self.logger.debug(f"🔈 Audio event from {uid}: {length} bytes")

                # Process for turn detection (independent of STT)
                if self.turn_detection and user:
                    user_id = user.user_id if hasattr(user, "user_id") else str(user)

                    # Process audio for turn detection
                    try:
                        await self.turn_detection.process_audio(
                            pcm,
                            user_id,
                            metadata={"timestamp": asyncio.get_event_loop().time()},
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Turn detection process_audio error: {e}", exc_info=True
                        )

                # Also process for STT if available
                if self.stt and user:
                    await self._handle_audio_input(pcm, user)

            except Exception as e:
                self.logger.error(f"Error handling audio received event: {e}")
                self.logger.error(traceback.format_exc())

        # Set up video track handler if image processors or video transformer are configured
        if self._connection:

            def on_track_added(track_id, track_type, user):
                user_id = user.user_id if user else "unknown"
                self.logger.info(
                    f"🎬 New track detected: {track_id} ({track_type}) from {user_id}"
                )
                # Handle video tracks
                if (
                    track_type == "video"
                    or getattr(track_type, "value", track_type)
                    == TrackType.TRACK_TYPE_VIDEO
                    or track_type == 2
                ) and (self.image_processors or self.video_transformer):
                    asyncio.create_task(
                        self._process_video_track(track_id, track_type, user)
                    )
                # Handle audio tracks for turn detection via unified interface if available
                elif self.turn_detection and (
                    track_type == "audio"
                    or getattr(track_type, "value", track_type)
                    == TrackType.TRACK_TYPE_AUDIO
                    or track_type == 1
                ):
                    # Turn detection will automatically track participants via process_audio
                    pass

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
        """Handle incoming audio data from Stream WebRTC connection for STT."""
        if not self.stt:
            return

        try:
            # STT processes all audio continuously for best quality

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
                # STT needs continuous audio stream for services like Deepgram
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
            # Check if it's the agent's turn to respond
            if self.turn_detection:
                # Use turn detection callbacks/events rather than polling detect_turn() with transcripts
                # The turn detection system will signal via callbacks when it's time to respond
                if not self._should_respond_based_on_turn_detection():
                    self.logger.debug(
                        f"Turn detection: Not agent's turn to respond to: {text[:50]}..."
                    )
                    return

            # Process with pre-processors
            processed_data = text
            for processor in self.pre_processors:
                processed_data = processor.process(processed_data)

            # Generate response using model
            if self.llm:
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
        if not self.llm:
            return ""

        try:
            # Create context with instructions and available tools
            context = f"""
            
            Available tools: {[str(tool) for tool in self.tools]}
            
            User input: {input_text}
            
            Respond appropriately based on your instructions.
            """

            response = await self.llm.generate(context)
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
        if self.llm:
            try:
                context = f"""
                
                You are joining a video call with {participant_count} participant(s).
                Generate a brief, friendly greeting to introduce yourself.
                Keep it under 2 sentences.
                """

                response = await self.llm.generate(context)
                return response
            except Exception as e:
                self.logger.error(f"Error generating greeting: {e}")
                return f"Hello everyone! I'm {self.name}"
        else:
            return f"Hello everyone! I'm {self.name}"

    async def _generate_participant_greeting(self, user_id: str) -> str:
        """Generate a greeting message for a new participant."""
        if self.llm:
            try:
                context = f"""
                
                A new participant (user-{user_id}) has joined the call.
                Generate a brief, friendly greeting to welcome them.
                Keep it under 2 sentences.
                """

                response = await self.llm.generate(context)
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

    def _setup_turn_detection_callbacks(self) -> None:
        """Set up turn detection callbacks using the event system."""
        try:
            # Set up standard turn detection events
            self.turn_detection.on(TurnEvent.TURN_STARTED.value, self._on_turn_started)
            self.turn_detection.on(TurnEvent.TURN_ENDED.value, self._on_turn_ended)
            self.turn_detection.on(
                TurnEvent.SPEECH_STARTED.value, self._on_speech_started_td
            )
            self.turn_detection.on(
                TurnEvent.SPEECH_ENDED.value, self._on_speech_ended_td
            )

        except Exception as e:
            self.logger.warning(f"Error setting up turn detection callbacks: {e}")

    def _should_respond_based_on_turn_detection(self) -> bool:
        """Check if the agent should respond based on current turn detection state."""
        try:
            # Check if turn detection is currently active
            return self.turn_detection.is_detecting()

        except Exception as e:
            self.logger.debug(
                f"Turn detection check error (defaulting to allow response): {e}"
            )
            return True  # Default to allowing response on error

    def _on_turn_started(self, event_data) -> None:
        """Handle when a participant starts their turn."""
        self.logger.info("Turn started - participant speaking")

    def _on_turn_ended(self, event_data) -> None:
        """Handle when a participant ends their turn."""
        self.logger.info("Turn ended - agent may respond")

    def _on_speech_started_td(self, event_data) -> None:
        """Handle speech start from turn detection."""
        self.logger.debug("Turn detection: Speech started")

    def _on_speech_ended_td(self, event_data) -> None:
        """Handle speech end from turn detection."""
        self.logger.debug("Turn detection: Speech ended")
