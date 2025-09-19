import asyncio
import logging
import traceback
from typing import Optional, List, Any, Union
from uuid import uuid4

import aiortc
from aiortc import VideoStreamTrack

from ..edge.types import Participant, PcmData, Connection, TrackType, User
from ..events.events import RealtimePartialTranscriptEvent
from ..llm.types import StandardizedTextDeltaEvent
from ..tts.tts import TTS
from ..stt.stt import STT
from ..vad import VAD
from ..events import STTTranscriptEvent, STTPartialTranscriptEvent, VADAudioEvent, RealtimeTranscriptEvent
from .reply_queue import ReplyQueue
from ..edge.edge_transport import EdgeTransport
from ..mcp import MCPBaseServer
from ..events import get_global_registry, EventType


from .conversation import StreamHandle, Message, Conversation
from ..llm.llm import LLM, LLMResponse
from ..llm.realtime import Realtime
from ..processors.base_processor import filter_processors, ProcessorType, BaseProcessor
from ..turn_detection import TurnEvent, TurnEventData, BaseTurnDetector
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from .agent_session import AgentSessionContextManager




class Agent:
    """
    Agent class makes it easy to build your own video AI.

    Commonly used methods

    * agent.join(call) // join a call
    * agent.llm.simple_response("greet the user")
    * await agent.finish() // (wait for the call session to finish)
    * agent.close() // cleanup

    TODO:
    - MCP functionality should be moved into its own class
    - Should edge be required?
    """
    def __init__(
        self,
        # edge network for video & audio
        edge: EdgeTransport,
        # llm, optionally with sts/realtime capabilities
        llm: LLM | Realtime,
        # the agent's user info
        agent_user: User,
        # instructions
        instructions: str = "Keep your replies short and dont use special characters.",
        # setup stt, tts, and turn detection if not using an llm with realtime/sts
        stt: Optional[STT] = None,
        tts: Optional[TTS] = None,
        turn_detection: Optional[BaseTurnDetector] = None,
        vad: Optional[VAD] = None,
        # for video gather data at an interval
        # - roboflow/ yolo typically run continuously
        # - often combined with API calls to fetch stats etc
        # - state from each processor is passed to the LLM
        processors: Optional[List[BaseProcessor]] = None,
        # MCP servers for external tool and resource access
        mcp_servers: Optional[List[MCPBaseServer]] = None,
    ):
        self.instructions = instructions
        self.edge = edge
        self.agent_user = agent_user

        self.logger = logging.getLogger(f"Agent[{self.agent_user.id}]")

        self.llm = llm
        self.stt = stt
        self.tts = tts
        self.turn_detection = turn_detection
        self.vad = vad
        self.processors = processors or []
        self.mcp_servers = mcp_servers or []
        self.queue = ReplyQueue(self)

        # we sync the user talking and the agent responses to the conversation
        # because we want to support streaming responses and can have delta updates for both
        # user and agent we keep an handle for both
        self.conversation: Optional[Conversation] = None
        self._user_conversation_handle: Optional[StreamHandle] = None
        self._agent_conversation_handle: Optional[StreamHandle] = None

        if self.llm is not None:
            self.llm._attach_agent(self)
            self.llm.on("after_llm_response", self._handle_after_response)
            self.llm.on('standardized.output_text.delta', self._handle_output_text_delta)

        # Initialize state variables
        self._is_running: bool = False
        self._current_frame = None
        self._interval_task = None
        self._callback_executed = False
        self._connection: Optional[Connection] = None
        self._audio_track: Optional[aiortc.AudioStreamTrack] = None
        self._video_track: Optional[VideoStreamTrack] = None
        self._sts_connection = None

        # validation time
        self._validate_configuration()

        self._prepare_rtc()
        self._setup_stt()
        self._setup_turn_detection()
        self._setup_vad()
        self._setup_mcp_servers()

    async def close(self):
        """Clean up all connections and resources."""
        self._is_running = False
        self._user_conversation_handle = None
        self._agent_conversation_handle = None

        # Disconnect from MCP servers
        await self._disconnect_mcp_servers()

        # Close Realtime connection
        if self._sts_connection:
            await self._sts_connection.__aexit__(None, None, None)
        self._sts_connection = None

        # Close RTC connection
        if self._connection:
            await self._connection.close()
        self._connection = None

        # Close STT
        if self.stt:
            await self.stt.close()

        # Close TTS
        if self.tts:
            await self.tts.close()

        # Stop turn detection
        if self.turn_detection:
            self.turn_detection.stop()

        # Stop audio track
        if self._audio_track:
            self._audio_track.stop()
        self._audio_track = None

        # Stop video track
        if self._video_track:
            self._video_track.stop()
        self._video_track = None

        # Cancel interval task
        if self._interval_task:
            self._interval_task.cancel()
        self._interval_task = None

        # Close edge transport
        self.edge.close()

    def on(self, event_type: EventType):
        #TODO: this approach is a bit ugly. also breaks with multiple agents.
        def decorator(func):
            registry = get_global_registry()
            registry.add_listener(event_type, func)
            return func
        return decorator

    async def join(self, call: Any) -> "AgentSessionContextManager":
        self.call = call
        self.conversation = None

        # Connect to MCP servers
        await self._connect_mcp_servers()

        # Only set up chat if we have LLM (for conversation capabilities)
        if self.llm:
            # ask the edge to start the chat
            self.conversation = self.edge.create_conversation(call, self.agent_user, self.instructions)

        # when using STS, we sync conversation using transcripts otherwise we fallback to ST (if available)
        # TODO: maybe agent.on(transcript?)
        if self.sts_mode:
            self.llm.on("transcript", self._on_transcript)
            self.llm.on("partial_transcript", self._on_partial_transcript)
        elif self.stt:
            self.stt.on("transcript", self._on_transcript)
            self.stt.on("partial_transcript", self._on_partial_transcript)

        """Join a Stream video call."""
        if self._is_running:
            raise RuntimeError("Agent is already running")

        self.logger.info(f"🤖 Agent joining call: {call.id}")


        # Ensure Realtime providers are ready before proceeding (they manage their own connection)
        if self.sts_mode and isinstance(self.llm, Realtime):
            await self.llm.connect()


        connection = await self.edge.join(self, call)
        self._connection = connection


        self._is_running = True

        registry = get_global_registry()
        #registry.add_connection_listeners(self._connection)

        self.logger.info(f"🤖 Agent joined call: {call.id}")

        # Set up audio and video tracks together to avoid SDP issues
        audio_track = self._audio_track if self.publish_audio else None
        video_track = self._video_track if self.publish_video else None

        if audio_track or video_track:
            await self.edge.publish_tracks(audio_track, video_track)

            # Set up event handlers for audio processing
            await self._listen_to_audio_and_video()

            # Realtime providers manage their own event loops; nothing to do here

            from .agent_session import AgentSessionContextManager

            return AgentSessionContextManager(self, self._connection)
        # In case tracks are not added, still return context manager
        from .agent_session import AgentSessionContextManager

        return AgentSessionContextManager(self, self._connection)


    async def finish(self):
        """Wait for the call to end gracefully."""
        # If connection is None or already closed, return immediately
        if not self._connection:
            logging.info("🔚 Agent connection already closed, finishing immediately")
            return

        try:
            fut = asyncio.get_event_loop().create_future()

            @self.edge.on("call_ended")
            def on_ended():
                if not fut.done():
                    fut.set_result(None)

            await fut
        except Exception as e:
            logging.warning(f"⚠️ Error while waiting for call to end: {e}")
            # Don't raise the exception, just log it and continue cleanup

    async def say(self, text):
        """
        Say exactly this
        """
        await self.queue.say_text(text, self.agent_user.id)

    async def create_user(self):
        response = await self.edge.create_user(self.agent_user)
        return response

    def _handle_output_text_delta(self, event: StandardizedTextDeltaEvent):
        """Handle partial LLM response text deltas."""

        if self.conversation is None:
            return

        self.logger.info(f"received standardized.output_text.delta {event}")
        # Create a new streaming message if we don't have one
        if self._agent_conversation_handle is None:
            self._agent_conversation_handle = self.conversation.start_streaming_message(
                role="assistant",
                user_id=self.agent_user.id,
                initial_content=event.delta,
            )
        else:
            self.conversation.append_to_message(self._agent_conversation_handle, event.delta)

    async def _handle_after_response(self, llm_response: LLMResponse):
        if self.conversation is None:
            return

        if self._agent_conversation_handle is None:
            message = Message(
                content=llm_response.text,
                role="assistant",
                user_id=self.agent_user.id,
            )
            self.conversation.add_message(message)
        else:
            self.conversation.complete_message(self._agent_conversation_handle)
            self._agent_conversation_handle = None

        # Resume the queue for TTS playback
        await self.queue.resume(llm_response, user_id=self.agent_user.id)

    def _setup_vad(self):
        if self.vad:
            self.logger.info("🎙️ Setting up VAD listeners")
            self.vad.on("audio", self._on_vad_audio)
            
    def _setup_mcp_servers(self):
        """Set up MCP servers if any are configured."""
        if self.mcp_servers:
            self.logger.info(f"🔌 Setting up {len(self.mcp_servers)} MCP server(s)")
            for i, server in enumerate(self.mcp_servers):
                self.logger.info(f"  {i+1}. {server.__class__.__name__}")
        else:
            self.logger.debug("No MCP servers configured")

    def _setup_turn_detection(self):
        if self.turn_detection:
            self.logger.info("🎙️ Setting up turn detection listeners")
            self.turn_detection.on(TurnEvent.TURN_STARTED.value, self._on_turn_started)
            self.turn_detection.on(TurnEvent.TURN_ENDED.value, self._on_turn_ended)
            self.turn_detection.start()

    def _setup_stt(self):
        if self.stt:
            self.logger.info("🎙️ Setting up STT event listeners")
            self.stt.on("error", self._on_stt_error)
            self._stt_setup = True
        else:
            self._stt_setup = False

    async def _listen_to_audio_and_video(self) -> None:
        # Handle audio data for STT or Realtime
        @self.edge.on("audio")
        async def on_audio_received(pcm: PcmData, participant: Participant):
            if self.turn_detection is not None:
                await self.turn_detection.process_audio(pcm, participant.user_id)

            await self._reply_to_audio(pcm, participant)

        # Always listen to remote video tracks so we can forward frames to Realtime providers
        @self.edge.on("track_added")
        async def on_track(track_id, track_type, user):
            asyncio.create_task(self._process_track(track_id, track_type, user))

    async def _reply_to_audio(
        self, pcm_data: PcmData, participant: Participant
    ) -> None:
        if participant and getattr(participant, "user_id", None) != self.agent_user.id:
            # first forward to processors
            try:
                # Extract audio bytes for processors using the proper PCM data structure
                # PCM data has: format, sample_rate, samples, pts, dts, time_base
                audio_bytes = pcm_data.samples.tobytes()
                if self.vad:
                    asyncio.create_task(self.vad.process_audio(pcm_data, participant))
                # Forward to audio processors (skip None values)
                for processor in self.audio_processors:
                    if processor is None:
                        continue
                    try:
                        await processor.process_audio(audio_bytes, participant.user_id)
                    except Exception as e:
                        self.logger.error(
                            f"Error in audio processor {type(processor).__name__}: {e}"
                        )

            except Exception as e:
                self.logger.error(f"Error processing audio for processors: {e}")

            # when in Realtime mode call the Realtime directly (non-blocking)
            if self.sts_mode and isinstance(self.llm, Realtime):
                task = asyncio.create_task(self.llm.send_audio_pcm(pcm_data))
                task.add_done_callback(lambda t: print(f"Task (send_audio_pcm) error: {t.exception()}"))
            else:
                # Process audio through STT
                if self.stt:
                    self.logger.debug(f"🎵 Processing audio from {participant}")
                    await self.stt.process_audio(pcm_data, participant)

    async def _process_track(self, track_id: str, track_type: str, participant):
        """
        - connect the track to video sender...
        -
        """

        """Process video frames from a specific track."""
        self.logger.info(
            f"🎥VDP: Processing track: {track_id} from user {getattr(participant, 'user_id', 'unknown')} (type: {track_type})"
        )

        # Only process video tracks - track_type might be string, enum or numeric (2 for video)
        if track_type != TrackType.TRACK_TYPE_VIDEO:
            self.logger.debug(f"Ignoring non-video track: {track_type}")
            return

        track = self.edge.add_track_subscriber(track_id)
        if not track:
            self.logger.error(f"❌ Failed to subscribe to track: {track_id}")
            return

        self.logger.info(
            f"✅ Successfully subscribed to video track: {track_id}, track object: {track}"
        )

        # Give the track a moment to be ready
        await asyncio.sleep(0.5)

        # If Realtime provider supports video, forward frames upstream once per track
        if self.sts_mode:

            try:
                # TODO: you don't always want this on :)
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

    def _on_vad_audio(self, event:VADAudioEvent):
        # bytes_to_pcm = bytes_to_pcm_data(event.audio_data)
        # asyncio.create_task(self.stt.process_audio(bytes_to_pcm, event.user_metadata))
        pass

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
        event: Union[STTPartialTranscriptEvent|RealtimePartialTranscriptEvent],
    ):
        self.logger.info(f"🎤 [Partial transcript]: {event.text}")

        if event.text and event.text.strip() and self.conversation:
            user_id = "unknown"
            if hasattr(event, "user_metadata"):
                user_id = getattr(event.user_metadata, "user_id", "unknown")

            # Check if we have an active handle for this user
            if self._user_conversation_handle is None:
                # Start a new streaming message for this user
                self._user_conversation_handle = self.conversation.start_streaming_message(
                    role="user",
                    user_id=user_id
                )

            # Append the partial transcript to the active message
            self.conversation.append_to_message(self._user_conversation_handle, event.text)

    async def _on_transcript(
        self, event: Union[STTTranscriptEvent|RealtimeTranscriptEvent]
    ):

        self.logger.info(f"🎤 [STT transcript]: {event.text}")

        # if the agent is in STS mode than we dont need to process the transcription
        if not self.sts_mode:
            await self._process_transcription(event.text, event.user_metadata)

        if self.conversation is None:
            return

        user_id = "unknown"
        if hasattr(event, "user_metadata"):
            user_id = getattr(event.user_metadata, "user_id", "unknown")

        if self._user_conversation_handle is None:
            # No partial transcripts were received, create a completed message directly
            message = Message(
                original=event,
                content=event.text,
                role="user",
                user_id=user_id,
            )
            self.conversation.add_message(message)
        else:
            # Replace with final text and complete the message
            self.conversation.replace_message(self._user_conversation_handle, event.text)
            self.conversation.complete_message(self._user_conversation_handle)
            self._user_conversation_handle = None

    async def _on_stt_error(self, error):
        """Handle STT service errors."""
        self.logger.error(f"❌ STT Error: {error}")

    async def _process_transcription(
        self, text: str, participant: Optional[Participant] = None
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
                # TODO: why is this different...? (48k framerate vs 16k below)
                self.logger.info("🎵 Using Realtime provider output track for audio")
            else:
                self._audio_track = self.edge.create_audio_track()
                if self.tts:
                    self.tts.set_output_track(self._audio_track)

        # Set up video track if video publishers are available
        if self.publish_video:
            # Get the first video publisher to create the track
            video_publisher = self.video_publishers[0]
            self._video_track = video_publisher.create_video_track()
            self.logger.info("🎥 Video track initialized from video publisher")

    async def _connect_mcp_servers(self):
        """Connect to all configured MCP servers."""
        if not self.mcp_servers:
            return
            
        self.logger.info(f"🔌 Connecting to {len(self.mcp_servers)} MCP server(s)")
        
        for i, server in enumerate(self.mcp_servers):
            try:
                self.logger.info(f"  Connecting to MCP server {i+1}/{len(self.mcp_servers)}: {server.__class__.__name__}")
                await server.connect()
                self.logger.info(f"  ✅ Connected to MCP server {i+1}/{len(self.mcp_servers)}")
            except Exception as e:
                self.logger.error(f"  ❌ Failed to connect to MCP server {i+1}/{len(self.mcp_servers)}: {e}")
                # Continue with other servers even if one fails
                
    async def _disconnect_mcp_servers(self):
        """Disconnect from all configured MCP servers."""
        if not self.mcp_servers:
            return
            
        self.logger.info(f"🔌 Disconnecting from {len(self.mcp_servers)} MCP server(s)")
        
        for i, server in enumerate(self.mcp_servers):
            try:
                self.logger.info(f"  Disconnecting from MCP server {i+1}/{len(self.mcp_servers)}: {server.__class__.__name__}")
                await server.disconnect()
                self.logger.info(f"  ✅ Disconnected from MCP server {i+1}/{len(self.mcp_servers)}")
            except Exception as e:
                self.logger.error(f"  ❌ Error disconnecting from MCP server {i+1}/{len(self.mcp_servers)}: {e}")
                # Continue with other servers even if one fails

    async def get_mcp_tools(self) -> List[Any]:
        """Get all available tools from all connected MCP servers."""
        tools = []
        
        for server in self.mcp_servers:
            if server.is_connected:
                try:
                    server_tools = await server.list_tools()
                    tools.extend(server_tools)
                except Exception as e:
                    self.logger.error(f"Error getting tools from MCP server {server.__class__.__name__}: {e}")
                    
        return tools
        
    async def call_mcp_tool(self, server_index: int, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on a specific MCP server.
        
        Args:
            server_index: Index of the MCP server in the mcp_servers list
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            
        Returns:
            The result of the tool call
        """
        if server_index >= len(self.mcp_servers):
            raise ValueError(f"Invalid server index: {server_index}")
            
        server = self.mcp_servers[server_index]
        if not server.is_connected:
            raise RuntimeError(f"MCP server {server_index} is not connected")
            
        return await server.call_tool(tool_name, arguments)

