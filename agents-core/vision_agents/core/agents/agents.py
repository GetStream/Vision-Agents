import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import uuid4

import aiortc
import getstream.models
from aiortc import VideoStreamTrack
from getstream.video.rtc import Call
from opentelemetry import trace
from opentelemetry.trace import Tracer

from ..edge.events import AudioReceivedEvent, CallEndedEvent, TrackAddedEvent
from ..edge.types import Connection, Participant, PcmData, TrackType, User
from ..events.manager import EventManager
from ..llm.events import (
    LLMResponseChunkEvent,
    LLMResponseCompletedEvent,
    RealtimePartialTranscriptEvent,
    RealtimeTranscriptEvent,
)
from ..llm.llm import LLM
from ..llm.realtime import Realtime
from ..logging_utils import CallContextToken, clear_call_context, set_call_context
from ..mcp import MCPBaseServer, MCPManager
from ..processors.base_processor import Processor, ProcessorType, filter_processors
from ..stt.events import STTPartialTranscriptEvent, STTTranscriptEvent
from ..stt.stt import STT
from ..tts.tts import TTS
from ..turn_detection import TurnDetector, TurnStartedEvent, TurnEndedEvent
from ..vad import VAD
from ..vad.events import VADAudioEvent
from . import events
from .conversation import Conversation, Message, StreamHandle

if TYPE_CHECKING:
    from vision_agents.plugins.getstream.stream_edge_transport import StreamEdge

    from .agent_session import AgentSessionContextManager

logger = logging.getLogger(__name__)


def _log_task_exception(task: asyncio.Task):
    try:
        task.result()
    except Exception:
        logger.exception("Error in background task")

class Agent:
    """
    Agent class makes it easy to build your own video AI.

    Example:

        # realtime mode
        agent = Agent(
            edge=getstream.Edge(),
            agent_user=agent_user,
            instructions="Read @voice-agent.md",
            llm=gemini.Realtime(),
            processors=[],  # processors can fetch extra data, check images/audio data or transform video
        )

    Commonly used methods

    * agent.join(call) // join a call
    * agent.llm.simple_response("greet the user")
    * await agent.finish() // (wait for the call session to finish)
    * agent.close() // cleanup

    Note: Don't reuse the agent object. Create a new agent object each time.
    """

    def __init__(
        self,
        # edge network for video & audio
        edge: "StreamEdge",
        # llm, optionally with sts/realtime capabilities
        llm: LLM | Realtime,
        # the agent's user info
        agent_user: User,
        # instructions
        instructions: str = "Keep your replies short and dont use special characters.",
        # setup stt, tts, and turn detection if not using a realtime llm
        stt: Optional[STT] = None,
        tts: Optional[TTS] = None,
        turn_detection: Optional[TurnDetector] = None,
        vad: Optional[VAD] = None,
        # for video gather data at an interval
        # - roboflow/ yolo typically run continuously
        # - often combined with API calls to fetch stats etc
        # - state from each processor is passed to the LLM
        processors: Optional[List[Processor]] = None,
        # MCP servers for external tool and resource access
        mcp_servers: Optional[List[MCPBaseServer]] = None,
        tracer: Tracer = trace.get_tracer("agents"),
    ):
        self.instructions = instructions
        self.edge = edge
        self.agent_user = agent_user

        # only needed in case we spin threads
        self._root_span = trace.get_current_span()
        self.tracer = tracer

        self.logger = logging.getLogger(f"Agent[{self.agent_user.id}]")

        self.events = EventManager()
        self.events.register_events_from_module(getstream.models, "call.")
        self.events.register_events_from_module(events)

        self.llm = llm
        self.stt = stt
        self.tts = tts
        self.turn_detection = turn_detection
        self.vad = vad
        self.processors = processors or []
        self.mcp_servers = mcp_servers or []
        self._call_context_token: CallContextToken | None = None

        # Initialize MCP manager if servers are provided
        self.mcp_manager = MCPManager(self.mcp_servers, self.llm, self.logger) if self.mcp_servers else None

        # we sync the user talking and the agent responses to the conversation
        # because we want to support streaming responses and can have delta updates for both
        # user and agent we keep an handle for both
        self.conversation: Optional[Conversation] = None
        self._user_conversation_handle: Optional[StreamHandle] = None
        self._agent_conversation_handle: Optional[StreamHandle] = None
        
        # Track pending transcripts for turn-based response triggering
        self._pending_user_transcripts: Dict[str, str] = {}

        # Merge plugin events BEFORE subscribing to any events
        for plugin in [stt, tts, turn_detection, vad, llm]:
            if plugin and hasattr(plugin, "events"):
                self.logger.info(f"Registered plugin {plugin}")
                self.events.merge(plugin.events)

        self.llm._attach_agent(self)
        self.llm.events.subscribe(self._handle_after_response)
        self.llm.events.subscribe(self._handle_output_text_delta)

        self.events.subscribe(self._on_vad_audio)
        self.events.subscribe(self._on_agent_say)
        # Initialize state variables
        self._is_running: bool = False
        self._current_frame = None
        self._interval_task = None
        self._callback_executed = False
        self._track_tasks : Dict[str, asyncio.Task] = {}
        self._connection: Optional[Connection] = None
        self._audio_track: Optional[aiortc.AudioStreamTrack] = None
        self._video_track: Optional[VideoStreamTrack] = None
        self._realtime_connection = None
        self._pc_track_handler_attached: bool = False

        # validation time
        self._validate_configuration()
        self._prepare_rtc()
        self._setup_stt()
        self._setup_turn_detection()

    async def simple_response(
        self, text: str, participant: Optional[Participant] = None
    ) -> None:
        """
        Overwrite simple_response if you want to change how the Agent class calls the LLM
        """
        with self.tracer.start_as_current_span("simple_response") as span:
            response = await self.llm.simple_response(
                text=text, processors=self.processors, participant=participant
            )
            span.set_attribute("text", text)
            span.set_attribute("response.text", response.text)
            span.set_attribute("response.original", response.original)

    def subscribe(self, function):
        """Subscribe a callback to the agent-wide event bus.

        The event bus is a merged stream of events from the edge, LLM, STT, TTS,
        VAD, and other registered plugins.

        Args:
            function: Async or sync callable that accepts a single event object.

        Returns:
            A disposable subscription handle (depends on the underlying emitter).
        """
        return self.events.subscribe(function)


    async def join(self, call: Call) -> "AgentSessionContextManager":
        # TODO: validation. join can only be called once
        with self.tracer.start_as_current_span("join"):
            if self._is_running:
                raise RuntimeError("Agent is already running")

            self.call = call
            self.conversation = None

            # Ensure all subsequent logs include the call context.
            self._set_call_logging_context(call.id)

            try:
                # Connect to MCP servers if manager is available
                if self.mcp_manager:
                    with self.tracer.start_as_current_span("mcp_manager.connect_all"):
                        await self.mcp_manager.connect_all()

                # Setup chat and connect it to transcript events
                self.conversation = await self.edge.create_conversation(
                    call, self.agent_user, self.instructions
                )
                self.events.subscribe(self._on_transcript)
                self.events.subscribe(self._on_partial_transcript)

                # Ensure Realtime providers are ready before proceeding (they manage their own connection)
                self.logger.info(f"🤖 Agent joining call: {call.id}")
                if isinstance(self.llm, Realtime):
                    await self.llm.connect()

                with self.tracer.start_as_current_span("edge.join"):
                    connection = await self.edge.join(self, call)
            except Exception:
                self._clear_call_logging_context()
                raise

            self._connection = connection
            self._is_running = True

            connection._connection._coordinator_ws_client.on_wildcard(
                "*", lambda event_name, event: self.events.send(event)
            )

            self.logger.info(f"🤖 Agent joined call: {call.id}")

            # Set up audio and video tracks together to avoid SDP issues
            audio_track = self._audio_track if self.publish_audio else None
            video_track = self._video_track if self.publish_video else None

            if audio_track or video_track:
                with self.tracer.start_as_current_span("edge.publish_tracks"):
                    await self.edge.publish_tracks(audio_track, video_track)

            self.logger.info(f"🤖 Agent joined call: {call.id}")

            # Listen to incoming tracks if any component needs them
            # This is independent of publishing - agents can listen without publishing
            # (e.g., STT-only agents that respond via text chat)
            if self._needs_audio_or_video_input():
                await self._listen_to_audio_and_video()

            from .agent_session import AgentSessionContextManager

            return AgentSessionContextManager(self, self._connection)

    async def finish(self):
        """Wait for the call to end gracefully.

        Subscribes to the edge transport's `call_ended` event and awaits it. If
        no connection is active, returns immediately.
        """
        # If connection is None or already closed, return immediately
        if not self._connection:
            logging.info("🔚 Agent connection already closed, finishing immediately")
            return

        try:
            fut = asyncio.get_event_loop().create_future()

            @self.edge.events.subscribe
            async def on_ended(event: CallEndedEvent):
                if not fut.done():
                    fut.set_result(None)

            await fut
        except Exception as e:
            logging.warning(f"⚠️ Error while waiting for call to end: {e}")
            # Don't raise the exception, just log it and continue cleanup

    async def close(self):
        """Clean up all connections and resources.

        Closes MCP connections, realtime output, active media tracks, processor
        tasks, the call connection, STT/TTS services, and stops turn detection.
        Safe to call multiple times.

        This is an async method because several components expose async shutdown
        hooks (e.g., WebRTC connections, plugin services).
        """
        self._is_running = False
        self._user_conversation_handle = None
        self._agent_conversation_handle = None
        self._clear_call_logging_context()

        # Disconnect from MCP servers
        if self.mcp_manager:
            await self.mcp_manager.disconnect_all()

        for processor in self.processors:
            processor.close()
        
        # Stop all video forwarders
        if hasattr(self, '_video_forwarders'):
            for forwarder in self._video_forwarders:
                try:
                    await forwarder.stop()
                except Exception as e:
                    self.logger.error(f"Error stopping video forwarder: {e}")
            self._video_forwarders.clear()

        # Close Realtime connection
        if self._realtime_connection:
            await self._realtime_connection.__aexit__(None, None, None)
        self._realtime_connection = None

        # shutdown task processing
        for _, track in self._track_tasks.items():
            track.cancel()

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

    # ------------------------------------------------------------------
    # Logging context helpers
    # ------------------------------------------------------------------
    def _set_call_logging_context(self, call_id: str) -> None:
        """Apply the call id to the logging context for the agent lifecycle."""

        if self._call_context_token is not None:
            self._clear_call_logging_context()
        self._call_context_token = set_call_context(call_id)

    def _clear_call_logging_context(self) -> None:
        """Remove the call id from the logging context if present."""

        if self._call_context_token is not None:
            clear_call_context(self._call_context_token)
            self._call_context_token = None

    async def create_user(self):
        """Create the agent user in the edge provider, if required.

        Returns:
            Provider-specific user creation response.
        """
        with self.tracer.start_as_current_span("edge.create_user"):
            return await self.edge.create_user(self.agent_user)

    async def _handle_output_text_delta(self, event: LLMResponseChunkEvent):
        """Handle partial LLM response text deltas."""

        if self.conversation is None:
            return

        self.logger.info(
            f"received standardized.output_text.delta {self._truncate_for_logging(event)}"
        )
        # Create a new streaming message if we don't have one
        if self._agent_conversation_handle is None:
            self._agent_conversation_handle = self.conversation.start_streaming_message(
                role="assistant",
                user_id=self.agent_user.id,
                initial_content=event.delta or "",
            )
        else:
            self.conversation.append_to_message(
                self._agent_conversation_handle, event.delta or ""
            )

    async def _handle_after_response(self, event: LLMResponseCompletedEvent):
        if self.conversation is None:
            return

        if event.text is None or not event.text.strip():
            return

        if self._agent_conversation_handle is None:
            message = Message(
                content=event.text,
                role="assistant",
                user_id=self.agent_user.id,
            )
            self.conversation.add_message(message)
        else:
            self.conversation.complete_message(self._agent_conversation_handle)
            self._agent_conversation_handle = None

        # Trigger TTS directly instead of through event system
        if self.tts and event.text and event.text.strip():
            await self.tts.send(event.text)

    def _on_vad_audio(self, event: VADAudioEvent):
        self.logger.info(f"Vad audio event {self._truncate_for_logging(event)}")

    def _on_rtc_reconnect(self):
        # update the code to listen?
        # republish the audio track and video track?
        # TODO: implement me
        pass

    async def _on_agent_say(self, event: events.AgentSayEvent):
        """Handle agent say events by calling TTS if available."""
        try:
            # Emit say started event
            synthesis_id = str(uuid4())
            self.events.send(
                events.AgentSayStartedEvent(
                    plugin_name="agent",
                    text=event.text,
                    user_id=event.user_id,
                    synthesis_id=synthesis_id,
                )
            )

            start_time = time.time()

            if self.tts is not None:
                # Call TTS with user metadata
                user_metadata = {"user_id": event.user_id}
                if event.metadata:
                    user_metadata.update(event.metadata)

                await self.tts.send(event.text, user_metadata)

                # Calculate duration
                duration_ms = (time.time() - start_time) * 1000

                # Emit say completed event
                self.events.send(
                    events.AgentSayCompletedEvent(
                        plugin_name="agent",
                        text=event.text,
                        user_id=event.user_id,
                        synthesis_id=synthesis_id,
                        duration_ms=duration_ms,
                    )
                )

                self.logger.info(f"Agent said: {event.text}")
            else:
                self.logger.warning("No TTS available, cannot synthesize speech")

        except Exception as e:
            # Emit say error event
            self.events.send(
                events.AgentSayErrorEvent(
                    plugin_name="agent",
                    text=event.text,
                    user_id=event.user_id,
                    error=e,
                )
            )
            self.logger.error(f"Error in agent say: {e}")

    async def say(self, text: str, user_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Make the agent say something using TTS.
        
        This is a convenience method that sends an AgentSayEvent to trigger TTS synthesis.
        
        Args:
            text: The text for the agent to say
            user_id: Optional user ID for the speech
            metadata: Optional metadata to include with the speech
        """
        self.events.send(events.AgentSayEvent(
            plugin_name="agent",
            text=text,
            user_id=user_id or self.agent_user.id,
            metadata=metadata
        ))

    def _setup_turn_detection(self):
        if self.turn_detection:
            self.logger.info("🎙️ Setting up turn detection listeners")
            self.events.subscribe(self._on_turn_event)
            self.turn_detection.start()

    def _setup_stt(self):
        if self.stt:
            self.logger.info("🎙️ Setting up STT event listeners")
            self.events.subscribe(self._on_stt_error)

    async def _listen_to_audio_and_video(self) -> None:
        # Handle audio data for STT or Realtime
        @self.edge.events.subscribe
        async def on_audio_received(event: AudioReceivedEvent):
            pcm = event.pcm_data
            participant = event.participant
            if not pcm or not participant:
                return

            if self.turn_detection is not None:
                await self.turn_detection.process_audio(pcm, participant.user_id)

            await self._reply_to_audio(pcm, participant)

        # Always listen to remote video tracks so we can forward frames to Realtime providers
        @self.edge.events.subscribe
        async def on_track(event: TrackAddedEvent):
            track_id = event.track_id
            track_type = event.track_type
            user = event.user
            if not track_id or not track_type:
                return

            task = asyncio.create_task(self._process_track(track_id, track_type, user))
            self._track_tasks[track_id] = task
            task.add_done_callback(_log_task_exception)

    async def _reply_to_audio(
        self, pcm_data: PcmData, participant: Participant
    ) -> None:
        if participant and getattr(participant, "user_id", None) != self.agent_user.id:
            # first forward to processors
            # Extract audio bytes for processors using the proper PCM data structure
            # PCM data has: format, sample_rate, samples, pts, dts, time_base
            audio_bytes = pcm_data.samples.tobytes()
            if self.vad:
                asyncio.create_task(self.vad.process_audio(pcm_data, participant))
            # Forward to audio processors (skip None values)
            for processor in self.audio_processors:
                if processor is None:
                    continue
                await processor.process_audio(audio_bytes, participant.user_id)


            # when in Realtime mode call the Realtime directly (non-blocking)
            if self.realtime_mode and isinstance(self.llm, Realtime):
                # TODO: this behaviour should be easy to change in the agent class
                asyncio.create_task(self.llm.simple_audio_response(pcm_data))
                #task.add_done_callback(lambda t: print(f"Task (send_audio_pcm) error: {t.exception()}"))
            # Process audio through STT
            elif self.stt:
                self.logger.debug(f"🎵 Processing audio from {participant}")
                await self.stt.process_audio(pcm_data, participant)

    async def _process_track(self, track_id: str, track_type: str, participant):
        # TODO: handle CancelledError
        # we only process video tracks
        if track_type != TrackType.TRACK_TYPE_VIDEO:
            return

        # subscribe to the video track
        track = self.edge.add_track_subscriber(track_id)
        if not track:
            self.logger.error(
                f"Failed to subscribe to {track_id}"
            )
            return

        # Import VideoForwarder
        from ..utils.video_forwarder import VideoForwarder
        
        # Create a SHARED VideoForwarder for the RAW incoming track
        # This prevents multiple recv() calls competing on the same track
        raw_forwarder = VideoForwarder(
            track,  # type: ignore[arg-type]
            max_buffer=30,
            fps=30,  # Max FPS for the producer (individual consumers can throttle down)
            name=f"raw_video_forwarder_{track_id}",
        )
        await raw_forwarder.start()
        self.logger.info("🎥 Created raw VideoForwarder for track %s", track_id)
        
        # Track forwarders for cleanup
        if not hasattr(self, '_video_forwarders'):
            self._video_forwarders = []
        self._video_forwarders.append(raw_forwarder)

        # If Realtime provider supports video, determine which track to send
        if self.realtime_mode:
            if self._video_track:
                # We have a video publisher (e.g., YOLO processor)
                # Create a separate forwarder for the PROCESSED video track
                self.logger.info("🎥 Forwarding PROCESSED video frames to Realtime provider")
                processed_forwarder = VideoForwarder(
                    self._video_track,  # type: ignore[arg-type]
                    max_buffer=30,
                    fps=30,
                    name=f"processed_video_forwarder_{track_id}",
                )
                await processed_forwarder.start()
                self._video_forwarders.append(processed_forwarder)
                
                if isinstance(self.llm, Realtime):
                    # Send PROCESSED frames with the processed forwarder
                    await self.llm._watch_video_track(self._video_track, shared_forwarder=processed_forwarder)
            else:
                # No video publisher, send raw frames
                self.logger.info("🎥 Forwarding RAW video frames to Realtime provider")
                if isinstance(self.llm, Realtime):
                    await self.llm._watch_video_track(track, shared_forwarder=raw_forwarder)


        hasImageProcessers = len(self.image_processors) > 0

        # video processors - pass the raw forwarder (they process incoming frames)
        for processor in self.video_processors:
            try:
                await processor.process_video(track, participant.user_id, shared_forwarder=raw_forwarder)
            except Exception as e:
                self.logger.error(
                    f"Error in video processor {type(processor).__name__}: {e}"
                )

        # Use raw forwarder for image processors - only if there are image processors
        if not hasImageProcessers:
            # No image processors, just keep the connection alive
            self.logger.info("No image processors, video processing handled by video processors only")
            return
        
        # Initialize error tracking counters
        timeout_errors = 0
        consecutive_errors = 0
        
        while True:
            try:
                # Use the raw forwarder instead of competing for track.recv()
                video_frame = await raw_forwarder.next_frame(timeout=2.0)

                if video_frame:
                    # Reset error counts on successful frame processing
                    timeout_errors = 0
                    consecutive_errors = 0

                    if hasImageProcessers:

                        img = video_frame.to_image()

                        for processor in self.image_processors:
                            try:
                                await processor.process_image(img, participant.user_id)
                            except Exception as e:
                                self.logger.error(
                                    f"Error in image processor {type(processor).__name__}: {e}"
                                )


                else:
                    self.logger.warning("🎥VDP: Received empty frame")
                    consecutive_errors += 1

            except asyncio.TimeoutError:
                # Exponential backoff for timeout errors
                timeout_errors += 1
                backoff_delay = min(2.0 ** min(timeout_errors, 5), 30.0)
                self.logger.debug(
                    f"🎥VDP: Applying backoff delay: {backoff_delay:.1f}s"
                )
                await asyncio.sleep(backoff_delay)
            except asyncio.CancelledError:
                return

            except Exception:
                raise

        # Cleanup and logging
        self.logger.info(f"🎥VDP: Video processing loop ended for track {track_id} - timeouts: {timeout_errors}, consecutive_errors: {consecutive_errors}")

    async def _on_turn_event(self, event: TurnStartedEvent | TurnEndedEvent) -> None:
        """Handle turn detection events."""
        # In realtime mode, the LLM handles turn detection, interruption, and responses itself
        if self.realtime_mode:
            return
        
        if isinstance(event, TurnStartedEvent):
            # Interrupt TTS when user starts speaking (barge-in)
            if event.speaker_id and event.speaker_id != self.agent_user.id:
                if self.tts:
                    self.logger.info(
                        f"👉 Turn started - interrupting TTS for participant {event.speaker_id}"
                    )
                    try:
                        await self.tts.stop_audio()
                    except Exception as e:
                        self.logger.error(f"Error stopping TTS: {e}")
                else:
                    self.logger.info(
                        f"👉 Turn started - participant speaking {event.speaker_id} : {event.confidence}"
                    )
            else:
                # Agent itself started speaking - this is normal
                self.logger.debug(
                    f"👉 Turn started - agent speaking {event.speaker_id}"
                )
        elif isinstance(event, TurnEndedEvent):
            self.logger.info(
                f"👉 Turn ended - participant {event.speaker_id} finished (duration: {event.duration}, confidence: {event.confidence})"
            )
            
            # When turn detection is enabled, trigger LLM response when user's turn ends
            # This is the signal that the user has finished speaking and expects a response
            if event.speaker_id and event.speaker_id != self.agent_user.id:
                # Get the accumulated transcript for this speaker
                transcript = self._pending_user_transcripts.get(event.speaker_id, "")
                
                if transcript and transcript.strip():
                    self.logger.info(f"🤖 Triggering LLM response after turn ended for {event.speaker_id}")
                    
                    # Create participant object if we have metadata
                    participant = None
                    if hasattr(event, 'custom') and event.custom:
                        # Try to extract participant info from custom metadata
                        participant = event.custom.get('participant')
                    
                    # Trigger LLM response with the complete transcript
                    if self.llm:
                        await self.simple_response(transcript, participant)
                    
                    # Clear the pending transcript for this speaker
                    self._pending_user_transcripts[event.speaker_id] = ""

    async def _on_partial_transcript(
        self, event: STTPartialTranscriptEvent | RealtimePartialTranscriptEvent
    ):
        self.logger.info(f"🎤 [Partial transcript]: {event.text}")

        if event.text and event.text.strip() and self.conversation:
            user_id = "unknown"
            if hasattr(event, "user_metadata"):
                user_id = getattr(event.user_metadata, "user_id", "unknown")

            # Check if we have an active handle for this user
            if self._user_conversation_handle is None:
                # Start a new streaming message for this user
                self._user_conversation_handle = (
                    self.conversation.start_streaming_message(
                        role="user", user_id=user_id
                    )
                )

            # Append the partial transcript to the active message
            self.conversation.append_to_message(
                self._user_conversation_handle, event.text
            )

    async def _on_transcript(self, event: STTTranscriptEvent | RealtimeTranscriptEvent):
        self.logger.info(f"🎤 [STT transcript]: {event.text}")

        if not event.text:
            return

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
            self.conversation.replace_message(
                self._user_conversation_handle, event.text
            )
            self.conversation.complete_message(self._user_conversation_handle)
            self._user_conversation_handle = None
        
        # In realtime mode, the LLM handles everything itself (STT, turn detection, responses)
        # Skip our manual LLM triggering logic
        if self.realtime_mode:
            return
        
        # Determine how to handle LLM triggering based on turn detection
        if self.turn_detection is not None:
            # With turn detection: accumulate transcripts and wait for TurnEndedEvent
            # Store/append the transcript for this user
            if user_id not in self._pending_user_transcripts:
                self._pending_user_transcripts[user_id] = event.text
            else:
                # Append to existing transcript (user might be speaking in chunks)
                self._pending_user_transcripts[user_id] += " " + event.text
            
            self.logger.debug(
                f"📝 Accumulated transcript for {user_id} (waiting for turn end): "
                f"{self._pending_user_transcripts[user_id][:100]}..."
            )
        else:
            # Without turn detection: trigger LLM immediately on transcript completion
            # This is the traditional STT -> LLM flow
            if self.llm:
                self.logger.info("🤖 Triggering LLM response immediately (no turn detection)")
                
                # Get participant from event metadata
                participant = None
                if hasattr(event, "user_metadata"):
                    participant = event.user_metadata
                
                await self.simple_response(event.text, participant)

    async def _on_stt_error(self, error):
        """Handle STT service errors."""
        self.logger.error(f"❌ STT Error: {error}")



    @property
    def realtime_mode(self) -> bool:
        """Check if the agent is in Realtime mode.

        Returns:
            True if `llm` is a `Realtime` implementation; otherwise False.
        """
        if self.llm is not None and isinstance(self.llm, Realtime):
            return True
        return False

    @property
    def publish_audio(self) -> bool:
        """Whether the agent should publish an outbound audio track.

        Returns:
            True if TTS is configured or when in Realtime mode.
        """
        if self.tts is not None or self.realtime_mode:
            return True
        return False

    @property
    def publish_video(self) -> bool:
        """Whether the agent should publish an outbound video track.
        """
        return len(self.video_publishers) > 0

    def _needs_audio_or_video_input(self) -> bool:
        """Check if agent needs to listen to incoming audio or video.

        This determines whether the agent should register listeners for incoming
        media tracks from other participants. This is independent of whether the
        agent publishes its own tracks.

        Returns:
            True if any component needs audio/video input from other participants.

        Examples:
            - Agent with STT but no TTS: needs_audio=True (listen-only agent)
            - Agent with audio processors: needs_audio=True (analysis agent)
            - Agent with video processors: needs_video=True (frame analysis)
            - Agent with only LLM and TTS: needs_audio=False (announcement bot)
        """
        # Audio input needed for:
        # - STT (for transcription)
        # - Audio processors (for audio analysis)
        # Note: VAD and turn detection are helpers for STT/TTS, not standalone consumers
        needs_audio = self.stt is not None or len(self.audio_processors) > 0

        # Video input needed for:
        # - Video processors (for frame analysis)
        # - Realtime mode with video (multimodal LLMs)
        needs_video = len(self.video_processors) > 0 or (
            self.realtime_mode and isinstance(self.llm, Realtime)
        )

        return needs_audio or needs_video

    @property
    def audio_processors(self) -> List[Any]:
        """Get processors that can process audio.

        Returns:
            List of processors that implement `process_audio(audio_bytes, user_id)`.
        """
        return filter_processors(self.processors, ProcessorType.AUDIO)

    @property
    def video_processors(self) -> List[Any]:
        """Get processors that can process video.

        Returns:
            List of processors that implement `process_video(track, user_id)`.
        """
        return filter_processors(self.processors, ProcessorType.VIDEO)

    @property
    def video_publishers(self) -> List[Any]:
        """Get processors capable of publishing a video track.

        Returns:
            List of processors that implement `create_video_track()`.
        """
        return filter_processors(self.processors, ProcessorType.VIDEO_PUBLISHER)

    @property
    def audio_publishers(self) -> List[Any]:
        """Get processors capable of publishing an audio track.

        Returns:
            List of processors that implement `create_audio_track()`.
        """
        return filter_processors(self.processors, ProcessorType.AUDIO_PUBLISHER)

    @property
    def image_processors(self) -> List[Any]:
        """Get processors that can process images.

        Returns:
            List of processors that implement `process_image()`.
        """
        return filter_processors(self.processors, ProcessorType.IMAGE)

    def _validate_configuration(self):
        """Validate the agent configuration."""
        if self.realtime_mode:
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
            if self.realtime_mode and isinstance(self.llm, Realtime):
                self._audio_track = self.llm.output_track
                self.logger.info("🎵 Using Realtime provider output track for audio")
            else:
                # TODO: what if we want to transform audio...
                # Get the required framerate and stereo setting from TTS plugin, default to 48000 for WebRTC
                if self.tts:
                    framerate = self.tts.get_required_framerate()
                    stereo = self.tts.get_required_stereo()
                else:
                    framerate = 48000
                    stereo = True  # Default to stereo for WebRTC
                self._audio_track = self.edge.create_audio_track(framerate=framerate, stereo=stereo)
                if self.tts:
                    self.tts.set_output_track(self._audio_track)

        # Set up video track if video publishers are available
        if self.publish_video:
            # Get the first video publisher to create the track
            video_publisher = self.video_publishers[0]
            # TODO: some lLms like moondream publish video
            self._video_track = video_publisher.publish_video_track()
            self.logger.info("🎥 Video track initialized from video publisher")


    def _truncate_for_logging(self, obj, max_length=200):
        """Truncate object string representation for logging to prevent spam."""
        obj_str = str(obj)
        if len(obj_str) > max_length:
            obj_str = obj_str[:max_length] + "... (truncated)"
        return obj_str