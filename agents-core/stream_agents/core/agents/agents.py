import asyncio
import logging
import traceback
from typing import Optional, List, Any
from uuid import uuid4

from aiortc import VideoStreamTrack

from ..tts.tts import TTS
from ..stt.stt import STT
from ..events import STTTranscriptEvent, STTPartialTranscriptEvent
from .reply_queue import ReplyQueue
from ..edge.edge_transport import EdgeTransport, StreamEdge
from getstream.chat.client import ChatClient
from getstream.models import User, ChannelInput, UserRequest
from getstream.video import rtc
from getstream.video.call import Call
from getstream.video.rtc import audio_track
from getstream.video.rtc.pb.stream.video.sfu.event import events_pb2
from getstream.video.rtc.pb.stream.video.sfu.models import models_pb2
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant
from getstream.video.rtc.track_util import PcmData
from getstream.video.rtc.tracks import (
    SubscriptionConfig,
    TrackSubscriptionConfig,
    TrackType,
)

from .conversation import StreamConversation
from ..llm.llm import LLM
from ..llm.realtime import Realtime
from ..processors.base_processor import filter_processors, ProcessorType, BaseProcessor
from ..turn_detection import TurnEvent, TurnEventData, BaseTurnDetector
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .agent_session import AgentSessionContextManager


class Agent:
    def __init__(
        self,
        # edge network for video & audio
        edge: Optional[EdgeTransport] = None,
        # llm, optionally with sts capabilities
        llm: Optional[LLM | Realtime] = None,
        # instructions
        instructions: str = "Keep your replies short and dont use special characters.",
        # setup stt, tts, and turn detection if not using an llm with realtime/sts
        stt: Optional[STT] = None,
        tts: Optional[TTS] = None,
        turn_detection: Optional[BaseTurnDetector] = None,
        # the agent's user info
        agent_user: Optional[UserRequest] = None,
        # for video gather data at an interval
        # - roboflow/ yolo typically run continuously
        # - often combined with API calls to fetch stats etc
        # - state from each processor is passed to the LLM
        processors: Optional[List[BaseProcessor]] = None,
    ):
        self.instructions = instructions
        if edge is None:
            edge = StreamEdge()
        self.edge = edge
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
        self.queue = ReplyQueue(self)
        self.conversation: Optional[StreamConversation] = None
        if self.llm is not None:
            self.llm.attach_agent(self)

        # Initialize state variables
        self._is_running: bool = False
        self._current_frame = None
        self._interval_task = None
        self._callback_executed = False
        self._connection: Optional[rtc.RTCConnection] = None
        self._audio_track: Optional[audio_track.AudioStreamTrack] = None
        self._video_track: Optional[VideoStreamTrack] = None
        self._sts_connection = None

        # validation time
        self._validate_configuration()

        self._prepare_rtc()
        self._setup_stt()
        self._setup_turn_detection()

    def before_response(self, input):
        pass

    async def after_response(self, llm_response):
        # In Realtime (STS) mode or when not joined to a call, conversation may be None.
        # Only resume the reply queue (which writes to conversation/tts) when a conversation exists.
        if self.conversation is not None:
            await self.queue.resume(llm_response)

    async def join(self, call: Call) -> "AgentSessionContextManager":
        self.call = call
        self.channel = None
        self.conversation = None

        # Only set up chat if we have LLM (for conversation capabilities)
        if self.llm:
            # TODO: I don't know the human user at this point in the code...
            chat_client: ChatClient = call.client.stream.chat
            self.channel = chat_client.get_or_create_channel(
                "videocall",
                call.id,
                data=ChannelInput(created_by_id=self.agent_user.id),
            )
            self.conversation = StreamConversation(
                self.instructions, [], self.channel.data.channel, chat_client
            )
            # mypy: set attr dynamically for both LLM and Realtime types
            setattr(self.llm, "conversation", self.conversation)
            # When using Realtime, mirror responses/transcripts into the conversation for UI updates
            if (
                self.sts_mode
                and hasattr(self.llm, "on")
                and self.conversation is not None
            ):

                @self.llm.on("response")  # type: ignore[attr-defined]
                async def _on_llm_response(event):
                    try:
                        text = (
                            getattr(event, "text", None)
                            if not isinstance(event, str)
                            else event
                        )
                        is_complete = getattr(event, "is_complete", True)
                        if text and self.conversation is not None:
                            if is_complete:
                                self.conversation.finish_last_message(text)
                            else:
                                self.conversation.partial_update_message(text, None)
                    except Exception as e:
                        self.logger.debug(
                            f"Error updating conversation from response: {e}"
                        )

                @self.llm.on("transcript")  # type: ignore[attr-defined]
                async def _on_llm_transcript(event):
                    try:
                        # Only mirror user transcripts; assistant transcripts will be reflected via response events
                        is_user = getattr(event, "is_user", True)
                        text = (
                            getattr(event, "text", None)
                            if not isinstance(event, str)
                            else event
                        )
                        if is_user and text and self.conversation is not None:
                            self.conversation.partial_update_message(text, None)
                    except Exception as e:
                        self.logger.debug(
                            f"Error updating conversation from transcript: {e}"
                        )

        """Join a Stream video call."""
        if self._is_running:
            raise RuntimeError("Agent is already running")

        self.logger.info(f"🤖 Agent joining call: {call.id}")

        if self.sts_mode:
            self.logger.info("🎤 Using Realtime (Speech-to-Speech) mode")
        else:
            self.logger.info("🎤 Using STT/TTS mode")

        # Ensure Realtime providers are ready before proceeding (they manage their own connection)
        if self.sts_mode and isinstance(self.llm, Realtime):
            await self.llm.wait_until_ready()

        # Traditional mode - use WebRTC connection
        # Configure subscription for audio and video
        subscription_config = SubscriptionConfig(
            default=self._get_subscription_config()
        )

        # Open RTC connection and keep it alive for the duration of the returned context manager
        connection_cm = await rtc.join(
            call, self.agent_user.id, subscription_config=subscription_config
        )
        connection = await connection_cm.__aenter__()
        self._connection = connection
        self._is_running = True

        self.logger.info(f"🤖 Agent joined call: {call.id}")

        # Set up audio and video tracks together to avoid SDP issues
        audio_track = self._audio_track if self.publish_audio else None
        video_track = self._video_track if self.publish_video else None

        if audio_track or video_track:
            await connection.add_tracks(audio=audio_track, video=video_track)
            if audio_track:
                self.logger.debug("🤖 Agent ready to speak")
            if video_track:
                self.logger.debug("🎥 Agent ready to publish video")

            # In Realtime mode we directly publish the provider's output track; no extra forwarding needed

            # Set up event handlers for audio processing
            await self._listen_to_audio_and_video()

            # Realtime providers manage their own event loops; nothing to do here

            from .agent_session import AgentSessionContextManager

            return AgentSessionContextManager(self, connection_cm)
        # In case tracks are not added, still return context manager
        from .agent_session import AgentSessionContextManager

        return AgentSessionContextManager(self, connection_cm)

    async def say(self, text):
        await self.queue.say_text(text)

    async def play_audio(self, pcm):
        await self.queue.send_audio(pcm)

    def _setup_turn_detection(self):
        if self.turn_detection:
            self.logger.info("🎙️ Setting up turn detection listeners")
            self.turn_detection.on(TurnEvent.TURN_STARTED.value, self._on_turn_started)
            self.turn_detection.on(TurnEvent.TURN_ENDED.value, self._on_turn_ended)
            self.turn_detection.start()

    def _setup_stt(self):
        if self.stt:
            self.logger.info("🎙️ Setting up STT event listeners")
            self.stt.on("transcript", self._on_transcript)
            self.stt.on("partial_transcript", self._on_partial_transcript)
            self.stt.on("error", self._on_stt_error)
            self._stt_setup = True
        else:
            self._stt_setup = False

    def _get_subscription_config(self):
        return TrackSubscriptionConfig(
            track_types=[
                TrackType.TRACK_TYPE_VIDEO,
                TrackType.TRACK_TYPE_AUDIO,
            ]
        )

    async def _listen_to_audio_and_video(self) -> None:
        """Set up event handlers for the connection."""
        if not self._connection:
            self.logger.error("❌ No active connections found")
            return

        # Handle new participants joining
        async def on_track_published(event: events_pb2.TrackPublished):
            try:
                self.logger.info(f"📢 Track published: {event}")
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

        # Handle audio data for STT or Realtime
        @self._connection.on("audio")
        async def on_audio_received(pcm: PcmData, participant: Participant):
            if not participant:
                import pdb

                pdb.set_trace()

            if self.turn_detection is not None:
                await self.turn_detection.process_audio(pcm, participant.user_id)

            await self.reply_to_audio(pcm, participant)

        # Always listen to remote video tracks so we can forward frames to Realtime providers
        @self._connection.on("track_added")
        async def on_track(track_id, track_type, user):
            self.logger.info("VDP: on track")
            asyncio.create_task(self._process_track(track_id, track_type, user))

    async def reply_to_audio(
        self, pcm_data: PcmData, participant: models_pb2.Participant
    ) -> None:
        if participant and getattr(participant, "user_id", None) != self.agent_user.id:
            # first forward to processors
            try:
                # TODO: remove this nonsense, we know its pcm
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

            # when in Realtime mode call the Realtime directly
            if self.sts_mode and isinstance(self.llm, Realtime):
                await self.llm.send_audio_pcm(pcm_data)
            else:
                # Process audio through STT
                if self.stt:
                    self.logger.debug(f"🎵 Processing audio from {participant}")
                    await self.stt.process_audio(pcm_data, participant)

    async def _process_track(self, track_id: str, track_type: str, participant):
        """Process video frames from a specific track."""
        self.logger.info(
            f"🎥VDP: Processing track: {track_id} from user {getattr(participant, 'user_id', 'unknown')} (type: {track_type})"
        )
        self.logger.info(
            f"🎥 Participant object: {participant}, type: {type(participant)}"
        )

        # Only process video tracks - track_type might be string, enum or numeric (2 for video)
        if track_type not in ("video", TrackType.TRACK_TYPE_VIDEO, 2):
            self.logger.debug(f"Ignoring non-video track: {track_type}")
            return

        self.logger.info(f"🎥 Processing VIDEO track: {track_id}")

        # Subscribe to the video track
        if self._connection is None:
            self.logger.error("❌ No active connection")
            return

        track = self._connection.subscriber_pc.add_track_subscriber(track_id)
        if not track:
            self.logger.error(f"❌ Failed to subscribe to track: {track_id}")
            return

        self.logger.info(
            f"✅ Successfully subscribed to video track: {track_id}, track object: {track}"
        )

        # Give the track a moment to be ready
        await asyncio.sleep(0.5)

        # If Realtime provider supports video, forward frames upstream once per track
        if self.sts_mode and isinstance(self.llm, Realtime):
            try:
                await self.llm.start_video_sender(track)
                self.logger.info("🎥 Forwarding video frames to Realtime provider")
            except Exception as e:
                self.logger.error(
                    f"Error starting video sender to Realtime provider: {e}"
                )

        hasImageProcessers = len(self.image_processors) > 0
        self.logger.info(
            f"📸 Has image processors: {hasImageProcessers}, count: {len(self.image_processors)}"
        )

        self.logger.info(
            f"📸 Starting video processing loop for track {track_id} {participant.user_id} {participant.name}"
        )
        self.logger.info(
            f"📸 Track readyState: {getattr(track, 'readyState', 'unknown')}"
        )
        self.logger.info(f"📸 Track kind: {getattr(track, 'kind', 'unknown')}")
        self.logger.info(f"📸 Track enabled: {getattr(track, 'enabled', 'unknown')}")
        self.logger.info(f"📸 Track muted: {getattr(track, 'muted', 'unknown')}")
        # Use the exact same pattern as the working example
        while True:
            try:
                video_frame = await asyncio.wait_for(track.recv(), timeout=5.0)
                if video_frame:
                    if hasImageProcessers:
                        img = video_frame.to_image()

                        for processor in self.image_processors:
                            try:
                                await processor.process_image(img, participant.user_id)
                            except Exception as e:
                                self.logger.error(
                                    f"Error in image processor {type(processor).__name__}: {e}"
                                )

                    # video processors
                    for processor in self.video_processors:
                        try:
                            await processor.process_video(track, participant.user_id)
                        except Exception as e:
                            self.logger.error(
                                f"Error in video processor {type(processor).__name__}: {e}"
                            )

            except Exception as e:
                # TODO: handle timouet differently, break on normal error
                self.logger.error(
                    f"📸 Error receiving track: {e} - {type(e)}, trying again"
                )
                await asyncio.sleep(0.5)

    def _on_turn_started(self, event_data: TurnEventData) -> None:
        """Handle when a participant starts their turn."""
        self.queue.pause()
        # todo(nash): If the participant starts speaking while TTS is streaming, we need to cancel it
        self.logger.info(
            f"👉 Turn started - participant speaking {event_data.speaker_id} : {event_data.confidence}"
        )

    def _on_turn_ended(self, event_data: TurnEventData) -> None:
        """Handle when a participant ends their turn."""
        self.logger.info(
            f"👉 Turn ended - participant {event_data.speaker_id} finished (duration: {event_data.confidence})"
        )

    async def _on_partial_transcript(
        self,
        event: STTPartialTranscriptEvent,
        participant: Participant = None,
        metadata=None,
    ):
        """Handle partial transcript from STT service."""
        if event.text and event.text.strip():
            if self.conversation:
                self.conversation.partial_update_message(event.text, participant)
            self.logger.debug(f"🎤 [partial]: {event.text}")

    async def _on_transcript(
        self, event: STTTranscriptEvent, participant: Participant = None, metadata=None
    ):
        """Handle final transcript from STT service."""
        if event.text and event.text.strip():
            if self.conversation:
                self.conversation.finish_last_message(event.text)

            # Process transcription through LLM and respond
            await self._process_transcription(event.text, participant)

    async def _on_stt_error(self, error):
        """Handle STT service errors."""
        print("HEY ERROR", error)
        self.logger.error(f"❌ STT Error: {error}")

    async def _process_transcription(
        self, text: str, participant: Participant = None
    ) -> None:
        if self.llm is not None:
            await self.llm.simple_response(
                text=text, processors=self.processors, participant=participant
            )

    @property
    def sts_mode(self) -> bool:
        """Check if the agent is in Realtime mode."""
        if self.llm is not None and isinstance(self.llm, Realtime):
            return True
        return False

    @property
    def publish_audio(self) -> bool:
        if self.tts is not None or self.sts_mode:
            return True
        else:
            return False

    @property
    def publish_video(self) -> bool:
        return len(self.video_publishers) > 0

    @property
    def audio_processors(self) -> List[Any]:
        """Get processors that can process audio."""
        return filter_processors(self.processors, ProcessorType.AUDIO)

    @property
    def video_processors(self) -> List[Any]:
        """Get processors that can process video."""
        return filter_processors(self.processors, ProcessorType.VIDEO)

    @property
    def video_publishers(self) -> List[Any]:
        """Get processors that can process video."""
        return filter_processors(self.processors, ProcessorType.VIDEO_PUBLISHER)

    @property
    def audio_publishers(self) -> List[Any]:
        """Get processors that can process video."""
        return filter_processors(self.processors, ProcessorType.AUDIO_PUBLISHER)

    @property
    def image_processors(self) -> List[Any]:
        """Get processors that can process images."""
        return filter_processors(self.processors, ProcessorType.IMAGE)

    def _validate_configuration(self):
        """Validate the agent configuration."""
        if self.sts_mode:
            # Realtime mode - should not have separate STT/TTS
            if self.stt or self.tts:
                self.logger.warning(
                    "Realtime mode detected: STT and TTS services will be ignored. "
                    "The Realtime model handles both speech-to-text and text-to-speech internally."
                )
                # Realtime mode - should not have separate STT/TTS
            if self.stt or self.turn_detection:
                self.logger.warning(
                    "Realtime mode detected: STT, TTS and Turn Detection services will be ignored. "
                    "The Realtime model handles both speech-to-text, text-to-speech and turn detection internally."
                )
        else:
            # Traditional mode - check if we have audio processing or just video processing
            has_audio_processing = self.stt or self.tts or self.turn_detection
            has_video_processing = any(
                hasattr(p, "process_video") or hasattr(p, "process_image")
                for p in self.processors
            )

            if has_audio_processing and not self.llm:
                raise ValueError(
                    "LLM is required when using audio processing (STT/TTS/Turn Detection)"
                )

            # Allow video-only mode without LLM
            if not has_audio_processing and not has_video_processing:
                raise ValueError(
                    "At least one processing capability (audio or video) is required"
                )

    def _prepare_rtc(self):
        # Variables are now initialized in __init__

        # Set up audio track if TTS is available
        if self.publish_audio:
            if self.sts_mode and isinstance(self.llm, Realtime):
                self._audio_track = self.llm.output_track
                self.logger.info("🎵 Using Realtime provider output track for audio")
            else:
                self._audio_track = audio_track.AudioStreamTrack(framerate=16000)
                if self.tts:
                    self.tts.set_output_track(self._audio_track)

        # Set up video track if video publishers are available
        if self.publish_video:
            # Get the first video publisher to create the track
            video_publisher = self.video_publishers[0]
            self._video_track = video_publisher.create_video_track()
            self.logger.info("🎥 Video track initialized from video publisher")

    async def _setup_sts_audio_forwarding(self, sts_connection, rtc_connection):
        """Set up audio forwarding from Realtime connection to WebRTC connection."""
        self.logger.info("🔗 Setting up Realtime audio forwarding")

        # Set up audio forwarding from Realtime to WebRTC
        if self._audio_track:

            async def forward_sts_audio(audio_data):
                """Forward audio from Realtime connection to WebRTC connection."""
                try:
                    self.logger.info(
                        f"🎵 Forwarding {len(audio_data)} bytes of Realtime audio to WebRTC"
                    )
                    # Send audio data to the audio track
                    if self._audio_track is not None:
                        await self._audio_track.send_audio(audio_data)
                    self.logger.debug("✅ Audio forwarded successfully")
                except Exception as e:
                    self.logger.error(f"Error forwarding Realtime audio: {e}")
                    self.logger.error(traceback.format_exc())

            # Check which type of Realtime connection we have
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
                self.logger.warning(
                    "⚠️ Realtime connection doesn't support audio forwarding"
                )
        else:
            self.logger.warning("⚠️ Audio track not available for forwarding")

    async def close(self):
        """Clean up all connections and resources."""
        self._is_running = False

        try:
            if self._sts_connection and hasattr(self._sts_connection, "__aexit__"):
                await self._sts_connection.__aexit__(None, None, None)
        except Exception as e:
            self.logger.debug(f"Error closing Realtime connection: {e}")
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

    async def finish(self):
        """Wait for the call to end gracefully."""
        # If connection is None or already closed, return immediately
        if not self._connection:
            logging.info("🔚 Agent connection already closed, finishing immediately")
            return

        try:
            fut = asyncio.get_event_loop().create_future()

            @self._connection.on("call_ended")
            def on_ended():
                if not fut.done():
                    fut.set_result(None)

            await fut
        except Exception as e:
            logging.warning(f"⚠️ Error while waiting for call to end: {e}")
            # Don't raise the exception, just log it and continue cleanup
