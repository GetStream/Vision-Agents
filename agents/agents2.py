import asyncio
import logging
import traceback
from contextlib import nullcontext
from typing import Optional, List, Protocol, Any
from uuid import uuid4

from getstream.models import User
from getstream.video import rtc
from getstream.video.rtc import audio_track
from getstream.video.rtc.pb.stream.video.sfu.event import events_pb2
from getstream.video.rtc.pb.stream.video.sfu.models import models_pb2
from getstream.video.rtc.tracks import (
    SubscriptionConfig,
    TrackSubscriptionConfig,
    TrackType,
)

from agents.agents import (
    TransformedVideoTrack,
    PreProcessor,
    TurnDetection,
    VideoTransformer,
    STT,
    TTS,
    LLM,
)
from processors.base_processor import filter_processors, ProcessorType
from turn_detection import TurnEvent, TurnEventData

"""
TODO
- Screenshot integration (once a second). Somehow video doesn't work. not sure why
- Reduce logging verbosity
- User a better pcm_data alternative. https://github.com/GetStream/stream-py/blob/webrtc/getstream/video/rtc/track_util.py#L17
- Yolo integration (like https://github.com/GetStream/video-ai-samples/blob/ab4913a6e07301de50f83e4bf2b7b376a375af99/live_sports_coach/kickboxing_example.py#L369)
"""


class Agent:
    def __init__(
        self,
        # llm, optionally with sts capabilities
        llm: Optional[LLM] = None,
        # setup stt, tts, and turn detection if not using realtime/sts
        stt: Optional[STT] = None,
        tts: Optional[TTS] = None,
        turn_detection: Optional[TurnDetection] = None,
        # the agent's user info
        agent_user: Optional[User] = None,
        # for video agents. gather data at an interval
        # - roboflow/ yolo typically run continuously
        # - often combined with API calls to fetch stats etc
        # - state from each processor is passed to the LLM
        processors: Optional[List[PreProcessor]] = None,
        # transformers dont keep state/ aren't relevant for the LLM
        # just for applying sound or video effects
        video_transformer: Optional[VideoTransformer] = None,
    ):
        # Create agent user if not provided
        if agent_user is None:
            agent_id = f"agent-{uuid4()}"
            # Create a basic User object with required parameters
            self.agent_user = User(
                id=agent_id,
                banned=False,
                online=True,
                role="user",
                custom={"name": "AI Agent"},
                teams_role={},
            )
        else:
            self.agent_user = agent_user

        self.logger = logging.getLogger(f"Agent[{self.agent_user.id}]")

        self.llm = llm
        self.stt = stt
        self.tts = tts
        self.turn_detection = turn_detection
        self.processors = processors or []
        self.video_transformer = video_transformer

        # validation time
        self.validate_configuration()
        self.prepare_rtc()
        self.setup_stt()
        self.create_user()
        self.setup_turn_detection()

    def setup_turn_detection(self):
        if self.turn_detection:
            self.logger.info("🎙️ Setting up turn detection listeners")
            self.turn_detection.on(TurnEvent.TURN_STARTED.value, self._on_turn_started)
            self.turn_detection.on(TurnEvent.TURN_ENDED.value, self._on_turn_started)

    def setup_stt(self):
        if self.stt:
            self.logger.info("🎙️ Setting up STT event listeners")
            self.stt.on("transcript", self._on_transcript)
            self.stt.on("partial_transcript", self._on_partial_transcript)
            self.stt.on("error", self._on_stt_error)
            self._stt_setup = True
        else:
            self._stt_setup = False

    async def say_text(self, text):
        await self.tts.send(text)

    async def play_audio(self, pcm):
        self._audio_track.send_audio(pcm)

    async def reply_to_text(self, input_text: str):
        """
        Receive text (from a transcription, or user input)
        Run it through the LLM, get a response. And reply
        """
        response = await self.llm.generate(input_text)
        await self.say_text(response)

    def get_subscription_config(self):
        return TrackSubscriptionConfig(
            track_types=[
                TrackType.TRACK_TYPE_VIDEO,
                #TrackType.TRACK_TYPE_AUDIO,
            ]
        )

    async def join(self, call) -> None:
        """Join a Stream video call."""
        if self._is_running:
            raise RuntimeError("Agent is already running")

        self.logger.info(f"🤖 Agent joining call: {call.id}")

        if self.sts_mode:
            self.logger.info("🎤 Using STS (Speech-to-Speech) mode")
        else:
            self.logger.info("🎤 Using traditional STT/TTS mode")

        try:
            stsContextManager = None
            if self.sts_mode:
                stsContextManager = await self.llm.connect(call, self.agent_user.id)

            # Traditional mode - use WebRTC connection
            # Configure subscription for audio and video
            subscription_config = SubscriptionConfig(
                default=self.get_subscription_config()
            )

            async with (
                await rtc.join(
                    call, self.agent_user.id, subscription_config=subscription_config
                ) as connection,
                stsContextManager or nullcontext() as stsConnection,
            ):
                self._connection = connection
                self._sts_connection = stsConnection
                self._is_running = True

                self.logger.info(f"🤖 Agent joined call: {call.id}")

                # Set up audio track if available
                if self.publish_audio:
                    await connection.add_tracks(audio=self._audio_track)
                    self.logger.debug("🤖 Agent ready to speak")

                # Set up video track if available
                if self.publish_video:
                    await connection.add_tracks(video=self._video_track)

                    self.logger.debug("🎥 Agent ready to publish video")


                # Set up STS audio forwarding if in STS mode
                if self.sts_mode and self._sts_connection:
                    self.logger.info("🎥 STS audio. Forward from openAI to Stream")
                    await self._setup_sts_audio_forwarding(stsConnection, connection)

                # Set up event handlers for audio processing
                await self.listen_to_audio_and_video()

                # listen to what the realtime model says
                if self.sts_mode:

                    async def process_sts_events():
                        try:
                            # TODO: some method to receive audio
                            async for event in stsConnection:
                                # also see https://platform.openai.com/docs/api-reference/realtime_server_events/input_audio_buffer/speech_stopped
                                # TODO: implement https://github.com/openai/openai-python/blob/main/examples/realtime/push_to_talk_app.py#L167
                                self.logger.debug(f"🔔 STS Event: {event.type}")
                                # Handle any STS-specific events here if needed
                        except Exception as e:
                            self.logger.error(f"❌ Error processing STS events: {e}")
                            self.logger.error(traceback.format_exc())

                    # Start STS event processing in background
                    sts_task = asyncio.create_task(process_sts_events())

                # Send initial greeting, if the LLM is configured to do so
                if self.llm:
                    self.llm.conversation_started(self)

                # Keep the agent running and listening
                self.logger.info("🎧 Agent is active - press Ctrl+C to stop")
                try:
                    # Wait for the connection to stay alive
                    await connection.wait()
                except Exception as e:
                    self.logger.error(f"❌ Error while waiting for connection: {e}")
                    self.logger.error(traceback.format_exc())

        except KeyboardInterrupt:
            self.logger.info("👋 Shutting down agent...")
        except Exception as e:
            self.logger.error(f"❌ Error during agent operation: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            # Use the comprehensive cleanup method
            try:
                await self.close()
            except Exception as e:
                self.logger.debug(f"Error during cleanup: {e}")

    async def listen_to_audio_and_video(self) -> None:
        """Set up event handlers for the connection."""
        if not self._connection:
            self.logger.error("❌ No active connections found")
            return

        # Handle new participants joining
        async def on_track_published(event: events_pb2.TrackPublished):
            try:

                self.logger.info(
                    f"📢 Track published: {event}"
                )
            except Exception as e:
                self.logger.error(f"❌ Error handling track published event: {e}")
                self.logger.error(traceback.format_exc())

        # Set up WebSocket event handlers
        try:
            if hasattr(self._connection, "_ws_client") and self._connection._ws_client:
                self._connection._ws_client.on_event(
                    "track_published", on_track_published
                )
        except Exception as e:
            self.logger.error(f"Error setting up WebSocket event handlers: {e}")

        # Handle audio data for STT or STS
        @self._connection.on("audio")
        async def on_audio_received(pcm, user):
            if self.turn_detection:
                await self.turn_detection.process_audio(pcm, user)

            await self.reply_to_audio(pcm, user)

        # listen to video tracks if we have video or image processors
        if self.video_processors or self.image_processors:

            @self._connection.on("track_added")
            async def on_track(track_id, track_type, user):
                asyncio.create_task(self._process_track(track_id, track_type, user))

    async def reply_to_audio(self, pcm_data, participant: models_pb2.Participant) -> None:
        if participant and participant != self.agent_user.id:
            # first forward to processors
            try:
                # Extract audio bytes for processors
                audio_bytes = None
                if hasattr(pcm_data, "samples"):
                    samples = pcm_data.samples
                    if hasattr(samples, "tobytes"):
                        audio_bytes = samples.tobytes()
                    else:
                        audio_bytes = samples.astype("int16").tobytes()
                elif isinstance(pcm_data, bytes):
                    audio_bytes = pcm_data
                else:
                    self.logger.debug(f"Unknown PCM data format: {type(pcm_data)}")
                    audio_bytes = pcm_data  # Try as-is

                # Forward to audio processors
                for processor in self.audio_processors:
                    try:
                        await processor.process_audio(audio_bytes, participant.user_id)
                    except Exception as e:
                        self.logger.error(
                            f"Error in audio processor {type(processor).__name__}: {e}"
                        )

            except Exception as e:
                self.logger.error(f"Error processing audio for processors: {e}")

            # when in STS mode call the STS directly
            if self.sts_mode:
                if hasattr(self.llm, "send_audio"):
                    await self.llm.send_audio(pcm_data, participant)
            else:
                # Process audio through STT
                if self.stt:
                    self.logger.debug(f"🎵 Processing audio from {participant}")
                    await self.stt.process_audio(pcm_data, participant)

    async def _process_track(self, track_id: str, track_type: str, participant):
        """Process video frames from a specific track."""
        self.logger.info(
            f"🎥 Processing track: {track_id} from user {getattr(participant, 'user_id', 'unknown')} (type: {track_type})"
        )
        self.logger.info(f"🎥 Participant object: {participant}, type: {type(participant)}")

        # Only process video tracks - track_type might be numeric (2 for video)
        if track_type != "video":
            self.logger.debug(f"Ignoring non-video track: {track_type}")
            return

        self.logger.info(f"🎥 Processing VIDEO track: {track_id}")

        # Subscribe to the video track
        track = self._connection.subscriber_pc.add_track_subscriber(track_id)
        if not track:
            self.logger.error(f"❌ Failed to subscribe to track: {track_id}")
            return
        
        self.logger.info(f"✅ Successfully subscribed to video track: {track_id}, track object: {track}")

        # Give the track a moment to be ready
        await asyncio.sleep(0.5)

        hasImageProcessers = len(self.image_processors) > 0
        self.logger.info(f"📸 Has image processors: {hasImageProcessers}, count: {len(self.image_processors)}")

        try:
            self.logger.info(f"📸 Starting video processing loop for track {track_id} {participant.user_id} {participant.name}")
            self.logger.info(f"📸 Track readyState: {getattr(track, 'readyState', 'unknown')}")
            self.logger.info(f"📸 Track kind: {getattr(track, 'kind', 'unknown')}")
            self.logger.info(f"📸 Track enabled: {getattr(track, 'enabled', 'unknown')}")
            self.logger.info(f"📸 Track muted: {getattr(track, 'muted', 'unknown')}")
            # Use the exact same pattern as the working example
            while True:
                try:
                    self.logger.info(f"📸 Blocking on track.recv")
                    video_frame = await asyncio.wait_for(track.recv(), timeout=5.0)
                    if video_frame:
                        self.logger.info(f"📸 Video frame received: {video_frame.time} - {video_frame.format}")
                        
                        if hasImageProcessers:
                            img = video_frame.to_image()
                            self.logger.info(f"📸 Converted to PIL Image: {img.size}")
                            
                            for processor in self.image_processors:
                                try:
                                    await processor.process_image(img, participant.user_id)
                                except Exception as e:
                                    self.logger.error(f"Error in image processor {type(processor).__name__}: {e}")
                        
                        # video processors
                        for processor in self.video_processors:
                            try:
                                await processor.process_video(track, participant.user_id)
                            except Exception as e:
                                self.logger.error(f"Error in video processor {type(processor).__name__}: {e}")
                                
                except Exception as e:
                    # TODO: handle timouet differently, break on normal error
                    self.logger.error(f"📸 Error receiving track: {e} - {type(e)}, trying again")
                    await asyncio.sleep(0.5)
        except Exception as e:
            self.logger.error(f"Fatal error in track processing {track_id}: {e}")
            self.logger.error(traceback.format_exc())

    def _on_turn_started(self, event_data: TurnEventData) -> None:
        """Handle when a participant starts their turn."""
        # todo(nash): If the participant starts speaking while TTS is streaming, we need to cancel it
        self.logger.info(f"👉 Turn started - participant speaking {event_data.speaker}")

    def _on_turn_ended(self, event_data: TurnEventData) -> None:
        """Handle when a participant ends their turn."""
        self.logger.info(f"👉 Turn ended - agent may respond {event_data.duration}")

    async def _on_partial_transcript(self, text: str, user=None, metadata=None):
        """Handle partial transcript from STT service."""
        if text and text.strip():
            user_info = user.user_id if user and hasattr(user, "user_id") else "unknown"
            self.logger.debug(f"🎤 [{user_info}] (partial): {text}")

    async def _on_transcript(self, text: str, user=None, metadata=None):
        """Handle final transcript from STT service."""
        if text and text.strip():
            user_info = user.user_id if user and hasattr(user, "user_id") else "unknown"
            self.logger.info(f"🎤 [{user_info}]: {text}")

            # Process transcription through LLM and respond
            await self._process_transcription(text, user)

    async def _on_stt_error(self, error):
        """Handle STT service errors."""
        self.logger.error(f"❌ STT Error: {error}")

    async def _process_transcription(self, text: str, user=None) -> None:
        await self.reply_to_text(text)

    @property
    def sts_mode(self) -> bool:
        """Check if the agent is in STS mode."""
        if self.llm is None:
            return False
        return getattr(self.llm, "sts", False)

    @property
    def publish_audio(self) -> bool:
        if self.tts is not None or self.sts_mode:
            return True
        else:
            return False

    @property
    def publish_video(self) -> bool:
        if self.video_transformer is not None:
            return True
        else:
            return False

    @property
    def audio_processors(self) -> List[Any]:
        """Get processors that can process audio."""
        return filter_processors(self.processors, ProcessorType.AUDIO)

    @property
    def video_processors(self) -> List[Any]:
        """Get processors that can process video."""
        return filter_processors(self.processors, ProcessorType.VIDEO)

    @property
    def image_processors(self) -> List[Any]:
        """Get processors that can process images."""
        return filter_processors(self.processors, ProcessorType.IMAGE)

    def validate_configuration(self):
        """Validate the agent configuration."""
        if self.sts_mode:
            # STS mode - should not have separate STT/TTS
            if self.stt or self.tts:
                self.logger.warning(
                    "STS mode detected: STT and TTS services will be ignored. "
                    "The STS model handles both speech-to-text and text-to-speech internally."
                )
                # STS mode - should not have separate STT/TTS
            if self.stt or self.turn_detection:
                self.logger.warning(
                    "STS mode detected: STT, TTS and Turn Detection services will be ignored. "
                    "The STS model handles both speech-to-text, text-to-speech and turn detection internally."
                )
        else:
            # Traditional mode - need LLM and either STT/TTS or both
            if not self.llm:
                raise ValueError("LLM is required for traditional mode")
            if not self.stt and not self.tts:
                raise ValueError(
                    "At least one of STT or TTS is required for traditional mode"
                )

    def prepare_rtc(self):
        # Initialize common variables
        self._current_frame = None
        self._interval_task = None
        self._is_running = False
        self._callback_executed = False

        self._connection: Optional[rtc.RTCConnection] = None
        self._audio_track: Optional[audio_track.AudioStreamTrack] = None
        self._video_track: Optional[TransformedVideoTrack] = None

        # Set up audio track if TTS is available
        if self.publish_audio:
            self._audio_track = audio_track.AudioStreamTrack(framerate=16000)
            if self.tts:
                self.tts.set_output_track(self._audio_track)

        # Set up video track if video transformer is available
        if self.publish_video:
            self._video_track = TransformedVideoTrack()
            self.logger.info("🎥 Video track initialized for transformation publishing")

    async def _setup_sts_audio_forwarding(self, sts_connection, rtc_connection):
        """Set up audio forwarding from STS connection to WebRTC connection."""
        self.logger.info("🔗 Setting up STS audio forwarding")

        # Set up audio forwarding from STS to WebRTC
        if self._audio_track:

            async def forward_sts_audio(audio_data):
                """Forward audio from STS connection to WebRTC connection."""
                try:
                    self.logger.info(
                        f"🎵 Forwarding {len(audio_data)} bytes of STS audio to WebRTC"
                    )
                    # Send audio data to the audio track
                    await self._audio_track.send_audio(audio_data)
                    self.logger.debug("✅ Audio forwarded successfully")
                except Exception as e:
                    self.logger.error(f"Error forwarding STS audio: {e}")
                    self.logger.error(traceback.format_exc())

            # Check which type of STS connection we have
            if hasattr(sts_connection, "on_audio") and callable(
                getattr(sts_connection, "on_audio")
            ):
                # Gemini Live-style connection - register audio callback
                sts_connection.on_audio(forward_sts_audio)
                self.logger.info("✅ Gemini Live audio forwarding configured")
            elif hasattr(sts_connection, "on_audio"):
                # OpenAI-style connection with on_audio decorator
                @sts_connection.on_audio
                async def forward_openai_audio(audio_data):
                    await forward_sts_audio(audio_data)

                self.logger.info("✅ OpenAI audio forwarding configured")
            else:
                self.logger.warning("⚠️ STS connection doesn't support audio forwarding")
        else:
            self.logger.warning("⚠️ Audio track not available for forwarding")

    async def close(self):
        """Clean up all connections and resources."""
        self._is_running = False

        try:
            if self._sts_connection and hasattr(self._sts_connection, "__aexit__"):
                await self._sts_connection.__aexit__(None, None, None)
        except Exception as e:
            self.logger.debug(f"Error closing STS connection: {e}")
        finally:
            self._sts_connection = None

        try:
            if self._connection and hasattr(self._connection, "__aexit__"):
                await self._connection.__aexit__(None, None, None)
        except Exception as e:
            self.logger.debug(f"Error closing RTC connection: {e}")
        finally:
            self._connection = None

        try:
            if self.stt and hasattr(self.stt, "close"):
                await self.stt.close()
        except Exception as e:
            self.logger.debug(f"Error closing STT: {e}")

        try:
            if self.tts and hasattr(self.tts, "close"):
                await self.tts.close()
        except Exception as e:
            self.logger.debug(f"Error closing TTS: {e}")
        try:
            if self.turn_detection and hasattr(self.turn_detection, "stop"):
                self.turn_detection.stop()
        except Exception as e:
                self.logger.debug(f"Error closing turn detection: {e}")

        try:
            if self._audio_track and hasattr(self._audio_track, "stop"):
                self._audio_track.stop()
        except Exception as e:
            self.logger.debug(f"Error stopping audio track: {e}")
        finally:
            self._audio_track = None

        try:
            if self._video_track and hasattr(self._video_track, "stop"):
                self._video_track.stop()
        except Exception as e:
            self.logger.debug(f"Error stopping video track: {e}")
        finally:
            self._video_track = None

        try:
            if self._interval_task:
                self._interval_task.cancel()
        except Exception as e:
            self.logger.debug(f"Error canceling interval task: {e}")
        finally:
            self._interval_task = None

    def create_user(self):
        """Create user - placeholder for any user setup logic."""
        pass
