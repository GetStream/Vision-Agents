import asyncio
import base64
import logging
import traceback
from contextlib import nullcontext
from typing import Optional, Callable, List, Protocol, Any
from uuid import uuid4

from getstream.models import User
from getstream.video import rtc
from getstream.video.rtc import audio_track
from getstream.video.rtc.tracks import SubscriptionConfig, TrackSubscriptionConfig, TrackType

from agents.agents import (
    TransformedVideoTrack,
    Tool,
    PreProcessor,
    TurnDetection,
    ImageProcessor,
    VideoTransformer,
    STT,
    TTS,
    VAD,
    STS, LLM
)

class Processor(Protocol):
    def start(self, data: Any) -> Any:
        """Any initial setup"""
        ...

    def receive_audio(self):
        pass

    def receive_video(self):
        pass

    def state(self):
        """return state for the llm"""
        pass

"""
TODO
- Params for setting up audio/video publish
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
                teams_role={}
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
        self.create_user()

    async def say_text(self, text):
        await self.tts.send(text)

    async def play_audio(self, pcm):
        self._audio_track.send_audio(pcm)

    async def reply_to_text(self, input_text: str):
        '''
        Receive text (from a transcription, or user input)
        Run it through the LLM, get a response. And reply
        '''
        response = await self.llm.generate(input_text)
        await self.say_text(response)

    async def reply_to_audio(self):
        pass

    def get_subscription_config(self):
        return TrackSubscriptionConfig(
            track_types=[
                TrackType.TRACK_TYPE_VIDEO,
                TrackType.TRACK_TYPE_AUDIO,
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

        # TODO: some property around if we are producing audio or not
        # TODO: isn't this already done in init???
        if self.tts or self.sts_mode:
            self._audio_track = audio_track.AudioStreamTrack(framerate=16000)
            if self.tts:
                self.tts.set_output_track(self._audio_track)

        # Set up video track if video transformer is available
        # TODO: some property around if we are producing video or not
        if self.video_transformer:
            self._video_track = TransformedVideoTrack()
            self.logger.info("🎥 Video track initialized for transformation publishing")

        try:
            stsContextManager = None
            if self.sts_mode:
                stsContextManager = await self.llm.connect(call, self.agent_user.id)


            # Traditional mode - use WebRTC connection
            # Configure subscription for audio and video
            #TODO: load from nice config/functions
            subscription_config = SubscriptionConfig(
                default=self.get_subscription_config()
            )

            async with await rtc.join(
                call, self.agent_user.id, subscription_config=subscription_config
            ) as connection, (stsContextManager or nullcontext()) as stsConnection:
                self._connection = connection
                self._sts_connection = stsConnection
                self._is_running = True

                self.logger.info(f"🤖 Agent joined call: {call.id}")



                # Set up audio track if available
                if self.publish_audio:
                    await connection.add_tracks(audio=self._audio_track)
                    self.logger.info("🤖 Agent ready to speak")

                # Set up video track if available
                if self.publish_video:
                    await connection.add_tracks(video=self._video_track)
                    self.logger.info("🎥 Agent ready to publish video")

                # Set up STS audio forwarding if in STS mode
                if self.sts_mode and self._sts_connection:
                    self.logger.info("🎥 STS audio. Forward from openAI to Stream")
                    await self._setup_sts_audio_forwarding(stsConnection, connection)

                # Set up event handlers for audio processing
                await self._setup_event_handlers()

                # Send initial greeting, if the LLM is configured to do so
                if self.llm and hasattr(self.llm, 'conversation_started'):
                    await self.llm.conversation_started(self)

                try:
                    self.logger.info("🎧 Agent is active - press Ctrl+C to stop")
                    if self.sts_mode:
                        # Process STS events in the background
                        async def process_sts_events():
                            try:
                                async for event in stsConnection:
                                    self.logger.debug(f"🔔 STS Event: {event.type}")
                                    # Handle any STS-specific events here if needed
                            except Exception as e:
                                self.logger.error(f"❌ Error processing STS events: {e}")
                                self.logger.error(traceback.format_exc())
                        
                        # Start STS event processing in background
                        sts_task = asyncio.create_task(process_sts_events())
                        
                        try:
                            await connection.wait()
                        finally:
                            sts_task.cancel()
                            try:
                                await sts_task
                            except asyncio.CancelledError:
                                pass
                    else:
                        await connection.wait()
                except Exception as e:
                    self.logger.error(f"❌ Error during agent operation: {e}")
                    self.logger.error(traceback.format_exc())
                    
        except Exception as e:
            self.logger.error(f"❌ Error during agent operation: {e}")
            self.logger.error(traceback.format_exc())
            raise
        finally:
            # Use the comprehensive cleanup method
            await self.close()

    async def _setup_event_handlers(self) -> None:
        """Set up event handlers for the connection."""
        if not self._connection:
            self.logger.error("❌ No active connections found")
            return

        # Handle new participants joining
        async def on_track_published(event):
            try:
                user_id = "unknown"
                if hasattr(event, "participant") and event.participant:
                    user_id = getattr(event.participant, "user_id", "unknown")

                track_id = getattr(event, "track_id", "unknown")
                track_type = getattr(event, "track_type", "unknown")

                self.logger.info(f"📢 Track published: {user_id} - {track_id} - {track_type}")

                if user_id and user_id != self.agent_user.id:
                    self.logger.info(f"👋 New participant joined: {user_id}")
                    
            except Exception as e:
                self.logger.error(f"❌ Error handling track published event: {e}")
                self.logger.error(traceback.format_exc())

        # Set up WebSocket event handlers
        try:
            if hasattr(self._connection, "_ws_client") and self._connection._ws_client:
                self._connection._ws_client.on_event("track_published", on_track_published)
        except Exception as e:
            self.logger.error(f"Error setting up WebSocket event handlers: {e}")

        # Handle audio data for STT or STS
        @self._connection.on("audio")
        async def on_audio_received(pcm, user):
            """Handle incoming audio data from participants."""
            try:
                if user and user != self.agent_user.id:
                    self.logger.debug(f"🎤 Received audio from {user}")
                    await self._handle_audio_input(pcm, user)
            except Exception as e:
                self.logger.error(f"Error handling audio received event: {e}")
                self.logger.error(traceback.format_exc())

    async def _handle_audio_input(self, pcm_data, user) -> None:
        """Handle incoming audio data from Stream WebRTC connection."""
        self.logger.info("Sending audio to STS from %s %s %s", self.sts_mode, self._sts_connection, hasattr(self._sts_connection, 'send_audio'))
        if self.sts_mode:
            # STS mode - send audio directly to STS connection
            if self._sts_connection and hasattr(self._sts_connection, 'connection'):
                try:
                    self.logger.debug(f"🎵 Sending audio to STS from {user}")
                    
                    # Extract audio data from PcmData object
                    self.logger.debug(f"PCM data type: {type(pcm_data)}")
                    
                    if hasattr(pcm_data, 'samples'):
                        # PcmData NamedTuple - extract samples (numpy array)
                        samples = pcm_data.samples
                        if hasattr(samples, 'tobytes'):
                            audio_bytes = samples.tobytes()
                        else:
                            # Convert numpy array to bytes
                            audio_bytes = samples.astype('int16').tobytes()
                    elif isinstance(pcm_data, bytes):
                        # Already bytes
                        audio_bytes = pcm_data
                    else:
                        self.logger.error(f"Unknown PCM data format: {type(pcm_data)}")
                        return
                    
                    # Encode audio as base64 for OpenAI realtime API
                    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                    
                    # Send as OpenAI realtime input_audio_buffer.append event
                    audio_event = {
                        "type": "input_audio_buffer.append",
                        "audio": audio_b64
                    }
                    
                    await self._sts_connection.connection.send(audio_event)
                except Exception as e:
                    self.logger.error(f"Error sending audio to STS from user {user}: {e}")
                    self.logger.error(traceback.format_exc())
            else:
                self.logger.debug("STS connection not available or doesn't support connection.send")
        else:
            # Traditional mode - use STT
            if not self.stt:
                self.logger.warning("No STT service available")
                return

            try:
                # Set up event listeners for transcription results (one-time setup)
                if not hasattr(self, "_stt_setup"):
                    self.logger.info("🎙️ Setting up STT event listeners")
                    self.stt.on("transcript", self._on_transcript)
                    self.stt.on("partial_transcript", self._on_partial_transcript)
                    self.stt.on("error", self._on_stt_error)
                    self._stt_setup = True

                # Process audio through STT
                self.logger.debug(f"🎵 Processing audio from {user}")
                await self.stt.process_audio(pcm_data, user)

            except Exception as e:
                self.logger.error(f"Error handling audio input from user {user}: {e}")
                self.logger.error(traceback.format_exc())

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
        """Process a complete transcription and generate response."""
        try:
            # Generate response using LLM
            if self.llm:
                # TODO: async version of this
                response = await self._generate_response(text)

                # Send response via TTS
                if self.tts and response:
                    try:
                        await self.tts.send(response)
                        self.logger.info(f"🤖 Responded: {response}")
                    except Exception as e:
                        self.logger.error(f"Error sending TTS response: {e}")
                        self.logger.error(traceback.format_exc())

        except Exception as e:
            self.logger.error(f"Error processing transcription: {e}")
            self.logger.error(traceback.format_exc())

    async def _generate_response(self, input_text: str) -> str:
        """Generate a response using the AI model."""
        try:
            response = await self.llm.generate(input_text)
            return response
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            self.logger.error(traceback.format_exc())
            return "I'm sorry, I encountered an error processing your request."

    @property
    def sts_mode(self) -> bool:
        """Check if the agent is in STS mode."""
        if self.llm is None:
            return False
        return getattr(self.llm, 'sts', False)
    
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
        
    

    def validate_configuration(self):
        """Validate the agent configuration."""
        if self.sts_mode:
            # STS mode - should not have separate STT/TTS
            if self.stt or self.tts:
                self.logger.warning(
                    "STS mode detected: STT and TTS services will be ignored. "
                    "The STS model handles both speech-to-text and text-to-speech internally."
                )
        else:
            # Traditional mode - need LLM and either STT/TTS or both
            if not self.llm:
                raise ValueError("LLM is required for traditional mode")
            if not self.stt and not self.tts:
                raise ValueError("At least one of STT or TTS is required for traditional mode")

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
        if hasattr(sts_connection, 'on_audio') and self._audio_track:
            @sts_connection.on_audio
            async def forward_sts_audio(audio_data):
                """Forward audio from STS connection to WebRTC connection."""
                try:
                    self.logger.debug("🎵 Forwarding STS audio to WebRTC")
                    # Send audio data to the audio track
                    await self._audio_track.send_audio(audio_data)
                except Exception as e:
                    self.logger.error(f"Error forwarding STS audio: {e}")
                    self.logger.error(traceback.format_exc())
            
            self.logger.info("✅ STS audio forwarding configured")
        else:
            self.logger.warning("⚠️ STS connection doesn't support audio forwarding or audio track not available")

    async def close(self):
        """Clean up all connections and resources."""
        self._is_running = False
        
        if self._sts_connection:
            await self._sts_connection.__aexit__(None, None, None)
            self._sts_connection = None
        
        if self._connection:
            await self._connection.__aexit__(None, None, None)
            self._connection = None
        
        if self.stt:
            await self.stt.close()
        
        if self.tts:
            await self.tts.close()
        
        if self._audio_track:
            self._audio_track.stop()
            self._audio_track = None
        
        if self._video_track:
            self._video_track.stop()
            self._video_track = None
        
        if self._interval_task:
            self._interval_task.cancel()
            self._interval_task = None

    def create_user(self):
        """Create user - placeholder for any user setup logic."""
        pass
