import asyncio
import logging
import traceback
from contextlib import nullcontext
from typing import Optional, List, Any
from uuid import uuid4

from aiortc import VideoStreamTrack
from openai.types.responses import EasyInputMessageParam, ResponseInputItemParam

from getstream.plugins.common import (
    TTS,
    STT,
    STTTranscriptEvent,
    STTPartialTranscriptEvent,
)
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

from .conversation import Conversation
from ..llm.llm import LLM
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
        llm: Optional[LLM] = None,
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

        # Initialize state variables
        self._is_running: bool = False
        self._current_frame = None
        self._interval_task = None
        self._callback_executed = False
        self._connection: Optional[rtc.RTCConnection] = None
        self._audio_track: Optional[audio_track.AudioStreamTrack] = None
        self._video_track: Optional[VideoStreamTrack] = None

        # validation time
        self._validate_configuration()

        self._prepare_rtc()
        self._setup_stt()
        self._setup_turn_detection()

    async def create_response(
        self,
        input: List[ResponseInputItemParam] | str,
        participant: Participant = None,
    ):
        # standardize on input
        if isinstance(input, str):
            if participant is not None:
                input = [
                    EasyInputMessageParam(content=input, role="user", type="message")
                ]
            else:
                input = [
                    EasyInputMessageParam(content=input, role="system", type="message")
                ]

        logging.info("participant in create response is %s", participant)
        if self.conversation:
            for i in input:
                if participant is not None:
                    user_id = participant.user_id
                else:
                    if i.get("role") == "assistant":
                        user_id = self.agent_user.id
                    else:
                        user_id = self.agent_user.id

                if i["type"] == "message":
                    content = i["content"]
                    if isinstance(content, str):
                        self.conversation.add_message(content, user_id)
                    else:
                        # Convert complex content to string representation
                        self.conversation.add_message(str(content), user_id)

        llm_response = await self.llm.simple_response(
            input, self.processors, conversation=self.conversation
        )
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
            self.conversation = Conversation(
                self.instructions, [], self.channel.data.channel, chat_client
            )

        """Join a Stream video call."""
        if self._is_running:
            raise RuntimeError("Agent is already running")

        self.logger.info(f"🤖 Agent joining call: {call.id}")

        if self.sts_mode:
            self.logger.info("🎤 Using STS (Speech-to-Speech) mode")
        else:
            self.logger.info("🎤 Using traditional STT/TTS mode")

        stsContextManager = None

        if self.sts_mode and self.llm is not None:
            stsContextManager = await self.llm.connect(call, self.agent_user.id)

        # Traditional mode - use WebRTC connection
        # Configure subscription for audio and video
        subscription_config = SubscriptionConfig(
            default=self._get_subscription_config()
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

            # Set up audio and video tracks together to avoid SDP issues
            audio_track = self._audio_track if self.publish_audio else None
            video_track = self._video_track if self.publish_video else None

            if audio_track or video_track:
                await connection.add_tracks(audio=audio_track, video=video_track)
                if audio_track:
                    self.logger.debug("🤖 Agent ready to speak")
                if video_track:
                    self.logger.debug("🎥 Agent ready to publish video")

            # Set up STS audio forwarding if in STS mode
            if self.sts_mode and self._sts_connection:
                self.logger.info("🎥 STS audio. Forward from openAI to Stream")
                await self._setup_sts_audio_forwarding(stsConnection, connection)

            # Set up event handlers for audio processing
            await self._listen_to_audio_and_video()

            # listen to what the realtime model says
            if self.sts_mode:

                async def process_sts_events():
                    try:
                        # TODO: some method to receive audio
                        if stsConnection is not None:
                            async for event in stsConnection:
                                # also see https://platform.openai.com/docs/api-reference/realtime_server_events/input_audio_buffer/speech_stopped
                                # TODO: implement https://github.com/openai/openai-python/blob/main/examples/realtime/push_to_talk_app.py#L167
                                self.logger.debug(f"🔔 STS Event: {event.type}")
                                # Handle any STS-specific events here if needed
                    except Exception as e:
                        self.logger.error(f"❌ Error processing STS events: {e}")
                        self.logger.error(traceback.format_exc())

                # Start STS event processing in background
                asyncio.create_task(process_sts_events())

            # Keep the agent running and listening
            self.logger.info("🎧 Agent is active - press Ctrl+C to stop")
            try:
                # Wait for the connection to stay alive
                await connection.wait()
            except Exception as e:
                self.logger.error(f"❌ Error while waiting for connection: {e}")
                self.logger.error(traceback.format_exc())

            from .agent_session import AgentSessionContextManager

            return AgentSessionContextManager(self)

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

        # Handle audio data for STT or STS
        @self._connection.on("audio")
        async def on_audio_received(pcm: PcmData, participant: Participant):
            if not participant:
                import pdb

                pdb.set_trace()

            if self.turn_detection is not None:
                await self.turn_detection.process_audio(pcm, participant.user_id)

            await self.reply_to_audio(pcm, participant)

        # listen to video tracks if we have video or image processors
        self.logger.info(
            "VDP: checking image and video processors %s %s",
            self.video_processors,
            self.image_processors,
        )
        if self.video_processors or self.image_processors:
            self.logger.info("VDP: ok image and video processors")

            @self._connection.on("track_added")
            async def on_track(track_id, track_type, user):
                self.logger.info("VDP: on track")
                asyncio.create_task(self._process_track(track_id, track_type, user))

    async def reply_to_audio(
        self, pcm_data: PcmData, participant: models_pb2.Participant
    ) -> None:
        if participant and participant != self.agent_user.id:
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

            # when in STS mode call the STS directly
            if self.sts_mode and self.llm is not None:
                if hasattr(self.llm, "send_audio"):
                    await self.llm.send_audio_pcm(pcm_data, participant)
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

        # Only process video tracks - track_type might be numeric (2 for video)
        if track_type != "video":
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
        await self.create_response(text, participant)

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
                    if self._audio_track is not None:
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
