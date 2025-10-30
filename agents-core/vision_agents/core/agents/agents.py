import asyncio
import contextlib
import logging
import tempfile
import time
import uuid
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import uuid4

import getstream.models
from aiortc import VideoStreamTrack
from getstream.video.rtc import Call

from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import TrackType

from ..edge import sfu_events
from ..edge.events import (
    AudioReceivedEvent,
    TrackAddedEvent,
    TrackRemovedEvent,
    CallEndedEvent,
)
from ..edge.types import Connection, Participant, PcmData, User, OutputAudioTrack
from ..events.manager import EventManager
from ..llm import events as llm_events
from ..llm.events import (
    LLMResponseChunkEvent,
    LLMResponseCompletedEvent,
    RealtimeUserSpeechTranscriptionEvent,
    RealtimeAgentSpeechTranscriptionEvent,
)
from ..llm.llm import LLM
from ..llm.realtime import Realtime
from ..mcp import MCPBaseServer, MCPManager
from ..processors.base_processor import Processor, ProcessorType, filter_processors
from ..stt.events import STTTranscriptEvent, STTErrorEvent
from ..stt.stt import STT
from ..tts.tts import TTS
from ..tts.events import TTSAudioEvent
from ..turn_detection import TurnDetector, TurnStartedEvent, TurnEndedEvent
from ..utils.logging import (
    CallContextToken,
    clear_call_context,
    set_call_context,
    configure_default_logging,
)
from ..utils.video_forwarder import VideoForwarder
from ..utils.video_utils import ensure_even_dimensions
from ..vad import VAD
from ..vad.events import VADAudioEvent
from . import events
from .conversation import Conversation
from dataclasses import dataclass
from opentelemetry.trace import set_span_in_context
from opentelemetry.trace.propagation import Span, Context
from opentelemetry import trace, context as otel_context
from opentelemetry.trace import Tracer
from opentelemetry.context import Token

if TYPE_CHECKING:
    from vision_agents.plugins.getstream.stream_edge_transport import StreamEdge

    from .agent_session import AgentSessionContextManager

logger = logging.getLogger(__name__)

tracer: Tracer = trace.get_tracer("agents")


def _log_task_exception(task: asyncio.Task):
    try:
        task.result()
    except Exception:
        logger.exception("Error in background task")


class _AgentLoggerAdapter(logging.LoggerAdapter):
    """
    A logger adapter to include the agent_id to the logs
    """

    def process(self, msg: str, kwargs):
        if self.extra:
            return "[Agent: %s] | %s" % (self.extra["agent_id"], msg), kwargs
        return super(_AgentLoggerAdapter, self).process(msg, kwargs)

# TODO: move me
@dataclass
class AgentOptions:
    model_dir: str

    def update(self, other: "AgentOptions") -> "AgentOptions":
        merged_dict = asdict(self)

        for key, value in asdict(other).items():
            if value is not None:
                merged_dict[key] = value

        return AgentOptions(**merged_dict)


def default_agent_options():
    return AgentOptions(model_dir=tempfile.gettempdir())


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
        options: Optional[AgentOptions] = None,
        tracer: Tracer = trace.get_tracer("agents"),
        # Configure the default logging for the sdk here. Pass None to leave the config intact.
        log_level: Optional[int] = logging.INFO,
    ):
        if log_level is not None:
            configure_default_logging(level=log_level)
        if options is None:
            options = default_agent_options()
        else:
            options = default_agent_options().update(options)
        self.options = options

        self.instructions = instructions
        self.edge = edge
        self.agent_user = agent_user
        self._agent_user_initialized = False

        # only needed in case we spin threads
        self.tracer = tracer
        self._root_span: Optional[Span] = None
        self._root_ctx: Optional[Context] = None

        self.logger = _AgentLoggerAdapter(logger, {"agent_id": self.agent_user.id})

        self.events = EventManager()
        self.events.register_events_from_module(getstream.models, "call.")
        self.events.register_events_from_module(events)
        self.events.register_events_from_module(sfu_events)
        self.events.register_events_from_module(llm_events)

        self.llm = llm
        self.stt = stt
        self.tts = tts
        self.turn_detection = turn_detection
        self.vad = vad
        self.processors = processors or []
        self.mcp_servers = mcp_servers or []
        self._call_context_token: CallContextToken | None = None
        self._context_token: Token[Context] | None = None

        # Initialize MCP manager if servers are provided
        self.mcp_manager = (
            MCPManager(self.mcp_servers, self.llm, self.logger)
            if self.mcp_servers
            else None
        )

        # we sync the user talking and the agent responses to the conversation
        # because we want to support streaming responses and can have delta updates for both
        # user and agent
        self.conversation: Optional[Conversation] = None

        # Track pending transcripts for turn-based response triggering
        self._pending_user_transcripts: Dict[str, str] = {}

        # Merge plugin events BEFORE subscribing to any events
        for plugin in [stt, tts, turn_detection, vad, llm, edge]:
            if plugin and hasattr(plugin, "events"):
                self.logger.debug(f"Register events from plugin {plugin}")
                self.events.merge(plugin.events)

        self.llm._attach_agent(self)

        self.events.subscribe(self._on_vad_audio)
        self.events.subscribe(self._on_agent_say)
        # Initialize state variables
        self._is_running: bool = False
        self._current_frame = None
        self._interval_task = None
        self._callback_executed = False
        self._track_tasks: Dict[str, asyncio.Task] = {}
        # Track metadata: track_id -> (track_type, participant, forwarder)
        self._active_video_tracks: Dict[str, tuple[int, Any, Any]] = {}
        self._video_forwarders: List[VideoForwarder] = []
        self._current_video_track_id: Optional[str] = None
        self._connection: Optional[Connection] = None
        self._audio_track: Optional[OutputAudioTrack] = None
        self._video_track: Optional[VideoStreamTrack] = None
        self._realtime_connection = None
        self._pc_track_handler_attached: bool = False

        # validation time
        self._validate_configuration()
        self._prepare_rtc()
        self._setup_stt()

    @contextlib.contextmanager
    def span(self, name):
        with tracer.start_as_current_span(name, context=self._root_ctx) as span:
            yield span

    def start_tracing(self):
        self._root_span = tracer.start_span("join")
        self._root_span.__enter__()
        self._root_ctx = set_span_in_context(self._root_span)
        # Activate the root context globally so all subsequent spans are nested under it
        self._context_token = otel_context.attach(self._root_ctx)

    def end_tracing(self):
        if self._root_span is not None:
            self._root_span.__exit__(None, None, None)
            self._root_span = None
            self._root_ctx = None
        # Detach the context token if it was set
        if hasattr(self, "_context_token") and self._context_token is not None:
            otel_context.detach(self._context_token)
            self._context_token = None

    def __aexit__(self, exc_type, exc_val, exc_tb):
        self.end_tracing()

    async def simple_response(
        self, text: str, participant: Optional[Participant] = None
    ) -> None:
        """
        Overwrite simple_response if you want to change how the Agent class calls the LLM
        """
        self.logger.info('🤖 Asking LLM to reply to "%s"', text)
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

    async def _setup_llm_events(self):
        @self.llm.events.subscribe
        async def on_llm_response_send_to_tts(event: LLMResponseCompletedEvent):
            # Trigger TTS directly instead of through event system
            if self.tts and event.text and event.text.strip():
                await self.tts.send(event.text)

        @self.llm.events.subscribe
        async def on_llm_response_sync_conversation(event: LLMResponseCompletedEvent):
            self.logger.info(f"🤖 [LLM response]: {event.text} {event.item_id}")

            if self.conversation is None:
                return

            with self.span("agent.on_llm_response_sync_conversation"):
                # Unified API: handles both streaming and non-streaming
                await self.conversation.upsert_message(
                    message_id=event.item_id,
                    role="assistant",
                    user_id=self.agent_user.id or "agent",
                    content=event.text or "",
                    completed=True,
                    replace=True,  # Replace any partial content from deltas
                )

        @self.llm.events.subscribe
        async def _handle_output_text_delta(event: LLMResponseChunkEvent):
            """Handle partial LLM response text deltas."""

            self.logger.info(
                f"🤖 [LLM delta response]: {event.delta} {event.item_id} {event.content_index}"
            )

            if self.conversation is None:
                return

            with self.span("agent._handle_output_text_delta"):
                await self.conversation.upsert_message(
                    message_id=event.item_id,
                    role="assistant",
                    user_id=self.agent_user.id or "agent",
                    content=event.delta or "",
                    content_index=event.content_index,
                    completed=False,  # Still streaming
                )

    async def _setup_speech_events(self):
        self.logger.info("_setup_speech_events")
        @self.events.subscribe
        async def on_error(event: STTErrorEvent):
            self.logger.error("stt error event %s", event)

        @self.events.subscribe
        async def on_stt_transcript_event_sync_conversation(event: STTTranscriptEvent):
            self.logger.info(f"🎤 [Transcript]: {event.text}")

            if self.conversation is None:
                return

            user_id = event.user_id()
            if user_id is None:
                raise ValueError("missing user_id")

            with self.span("agent.on_stt_transcript_event_sync_conversation"):
                await self.conversation.upsert_message(
                    message_id=str(uuid.uuid4()),
                    role="user",
                    user_id=user_id,
                    content=event.text or "",
                    completed=True,
                    replace=True,  # Replace any partial transcripts
                    original=event,
                )

        @self.events.subscribe
        async def on_realtime_user_speech_transcription(
            event: RealtimeUserSpeechTranscriptionEvent,
        ):
            self.logger.info(f"🎤 [User transcript]: {event.text}")

            if self.conversation is None or not event.text:
                return

            with self.span("agent.on_realtime_user_speech_transcription"):
                await self.conversation.upsert_message(
                    message_id=str(uuid.uuid4()),
                    role="user",
                    user_id=event.user_id() or "",
                    content=event.text,
                    completed=True,
                    replace=True,
                    original=event,
                )

        @self.events.subscribe
        async def on_realtime_agent_speech_transcription(
            event: RealtimeAgentSpeechTranscriptionEvent,
        ):
            self.logger.info(f"🎤 [Agent transcript]: {event.text}")

            if self.conversation is None or not event.text:
                return

            with self.span("agent.on_realtime_agent_speech_transcription"):
                await self.conversation.upsert_message(
                    message_id=str(uuid.uuid4()),
                    role="assistant",
                    user_id=self.agent_user.id or "",
                    content=event.text,
                    completed=True,
                    replace=True,
                    original=event,
                )

        @self.events.subscribe
        async def _on_tts_audio_write_to_output(event: TTSAudioEvent):
            if self._audio_track and event and event.audio_data is not None:
                await self._audio_track.write(event.audio_data)

        @self.events.subscribe
        async def on_stt_transcript_event_create_response(event: STTTranscriptEvent):
            if self.realtime_mode or not self.llm:
                # when running in realtime mode, there is no need to send the response to the LLM
                return

            user_id = event.user_id()
            if user_id is None:
                raise ValueError("user id is none, this indicates a bug in the code")

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
                with self.span("agent.on_stt_transcript_event_create_response"):
                    await self.simple_response(event.text, event.participant)

    async def join(self, call: Call) -> "AgentSessionContextManager":
        # TODO: validation. join can only be called once
        self.logger.info("joining call")
        if self.stt:
            # TODO: run this in parallel for various services?
            await self.stt.start()
        self.start_tracing()

        if self._root_span:
            self._root_span.set_attribute("call_id", call.id)
            if self.agent_user.id:
                self._root_span.set_attribute("agent_id", self.agent_user.id)

        if self._is_running:
            raise RuntimeError("Agent is already running")

        await self.create_user()
        await self._setup_turn_detection()

        self.call = call
        self.conversation = None

        # Ensure all subsequent logs include the call context.
        self._set_call_logging_context(call.id)

        # Setup chat and connect it to transcript events (we'll wait at the end)
        create_conversation_coro = self.edge.create_conversation(
            call, self.agent_user, self.instructions
        )

        await self._setup_llm_events()
        await self._setup_speech_events()

        try:
            # Connect to MCP servers if manager is available
            if self.mcp_manager:
                with self.span("mcp_manager.connect_all"):
                    await self.mcp_manager.connect_all()

            # Ensure Realtime providers are ready before proceeding (they manage their own connection)
            self.logger.info(f"🤖 Agent joining call: {call.id}")
            if isinstance(self.llm, Realtime):
                await self.llm.connect()

            with self.span("edge.join"):
                connection = await self.edge.join(self, call)
        except Exception:
            self.clear_call_logging_context()
            raise

        self._connection = connection
        self._is_running = True

        self.logger.info(f"🤖 Agent joined call: {call.id}")

        # Set up audio and video tracks together to avoid SDP issues
        audio_track = self._audio_track if self.publish_audio else None
        video_track = self._video_track if self.publish_video else None

        if audio_track or video_track:
            with self.span("edge.publish_tracks"):
                await self.edge.publish_tracks(audio_track, video_track)

        connection._connection._coordinator_ws_client.on_wildcard(
            "*",
            lambda event_name, event: self.events.send(event),
        )

        connection._connection._ws_client.on_wildcard(
            "*",
            lambda event_name, event: self.events.send(event),
        )

        # Listen to incoming tracks if any component needs them
        # This is independent of publishing - agents can listen without publishing
        # (e.g., STT-only agents that respond via text chat)
        if self._needs_audio_or_video_input():
            await self._listen_to_audio_and_video()

        from .agent_session import AgentSessionContextManager

        # wait for conversation creation coro at the very end of the join flow
        self.conversation = await create_conversation_coro
        return AgentSessionContextManager(self, self._connection)

    async def finish(self):
        """Wait for the call to end gracefully.
        Subscribes to the edge transport's `call_ended` event and awaits it. If
        no connection is active, returns immediately.
        """
        # If connection is None or already closed, return immediately
        if not self._connection:
            self.logger.info(
                "🔚 Agent connection is already closed, finishing immediately"
            )
            return


        with self.span("agent.finish"):
            # If connection is None or already closed, return immediately
            if not self._connection:
                logging.info(
                    "🔚 Agent connection already closed, finishing immediately"
                )
                return

            @self.edge.events.subscribe
            async def on_ended(event: CallEndedEvent):
                self._is_running = False

            while self._is_running:
                try:
                    await asyncio.sleep(0.0001)
                except asyncio.CancelledError:
                    self._is_running = False

            await asyncio.shield(self.close())

    async def close(self):
        """Clean up all connections and resources.

        Closes MCP connections, realtime output, active media tracks, processor
        tasks, the call connection, STT/TTS services, and stops turn detection.
        Safe to call multiple times.

        This is an async method because several components expose async shutdown
        hooks (e.g., WebRTC connections, plugin services).
        """
        self.end_tracing()
        self._is_running = False
        self.clear_call_logging_context()

        # Disconnect from MCP servers
        if self.mcp_manager:
            await self.mcp_manager.disconnect_all()

        for processor in self.processors:
            processor.close()

        # Stop all video forwarders
        if hasattr(self, "_video_forwarders"):
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
            await self.turn_detection.stop()

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
            self.clear_call_logging_context()
        self._call_context_token = set_call_context(call_id)

    def clear_call_logging_context(self) -> None:
        """Remove the call id from the logging context if present."""

        if self._call_context_token is not None:
            clear_call_context(self._call_context_token)
            self._call_context_token = None

    async def create_user(self) -> None:
        """Create the agent user in the edge provider, if required."""

        if self._agent_user_initialized:
            return None

        with self.span("edge.create_user"):
            if not self.agent_user.id:
                self.agent_user.id = f"agent-{uuid4()}"
            await self.edge.create_user(self.agent_user)
            self._agent_user_initialized = True

        return None

    def _on_vad_audio(self, event: VADAudioEvent):
        self.logger.debug(f"Vad audio event {self._truncate_for_logging(event)}")

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

                self.logger.info(f"🤖 Agent said: {event.text}")
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

    async def say(
        self,
        text: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Make the agent say something using TTS.

        This is a convenience method that sends an AgentSayEvent to trigger TTS synthesis.

        Args:
            text: The text for the agent to say
            user_id: Optional user ID for the speech
            metadata: Optional metadata to include with the speech
        """
        self.events.send(
            events.AgentSayEvent(
                plugin_name="agent",
                text=text,
                user_id=user_id or self.agent_user.id,
                metadata=metadata,
            )
        )

        if self.conversation is not None:
            await self.conversation.upsert_message(
                role="assistant",
                user_id=user_id or self.agent_user.id or "agent",
                content=text,
                completed=True,
            )

    async def _setup_turn_detection(self):
        if self.turn_detection:
            self.logger.info("🎙️ Setting up turn detection listeners")
            self.events.subscribe(self._on_turn_event)
            await self.turn_detection.start()

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

            if self.turn_detection is not None and participant is not None:
                await self.turn_detection.process_audio(
                    pcm, participant, conversation=self.conversation
                )

            if participant is not None:
                await self._reply_to_audio(pcm, participant)

        # Always listen to remote video tracks so we can forward frames to Realtime providers
        @self.edge.events.subscribe
        async def on_track(event: TrackAddedEvent):
            track_id = event.track_id
            track_type = event.track_type
            user = event.user
            if not track_id or not track_type:
                return

            # If track is already being processed, just switch to it
            if track_id in self._active_video_tracks:
                track_type_name = TrackType.Name(track_type)
                self.logger.info(
                    f"🎥 Track re-added: {track_type_name} ({track_id}), switching to it"
                )

                if self.realtime_mode and isinstance(self.llm, Realtime):
                    # Get the existing forwarder and switch to this track
                    _, _, forwarder = self._active_video_tracks[track_id]
                    track = self.edge.add_track_subscriber(track_id)
                    if track and forwarder:
                        await self.llm._watch_video_track(
                            track, shared_forwarder=forwarder
                        )
                        self._current_video_track_id = track_id
                return

            task = asyncio.create_task(self._process_track(track_id, track_type, user))
            self._track_tasks[track_id] = task
            task.add_done_callback(_log_task_exception)

        @self.edge.events.subscribe
        async def on_track_removed(event: TrackRemovedEvent):
            track_id = event.track_id
            track_type = event.track_type
            if not track_id:
                return

            track_type_name = TrackType.Name(track_type) if track_type else "unknown"
            self.logger.info(f"🎥 Track removed: {track_type_name} ({track_id})")

            # Cancel the processing task for this track
            if track_id in self._track_tasks:
                self._track_tasks[track_id].cancel()
                self._track_tasks.pop(track_id)

            # Clean up track metadata
            self._active_video_tracks.pop(track_id, None)

            # If this was the active track, switch to any other available track
            if (
                track_id == self._current_video_track_id
                and self.realtime_mode
                and isinstance(self.llm, Realtime)
            ):
                self.logger.info(
                    "🎥 Active video track removed, switching to next available"
                )
                self._current_video_track_id = None
                await self._switch_to_next_available_track()

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

            for processor in self.audio_processors:
                if processor is None:
                    continue
                asyncio.create_task(
                    processor.process_audio(audio_bytes, participant.user_id)
                )

            # when in Realtime mode call the Realtime directly (non-blocking)
            if self.realtime_mode and isinstance(self.llm, Realtime):
                # TODO: this behaviour should be easy to change in the agent class
                asyncio.create_task(
                    self.llm.simple_audio_response(pcm_data, participant)
                )

            # Process audio through STT
            elif self.stt:
                self.logger.debug(f"🎵 Processing audio from {participant}")
                asyncio.create_task(self.stt.process_audio(pcm_data, participant))

    async def _switch_to_next_available_track(self) -> None:
        """Switch to any available video track."""
        if not self._active_video_tracks:
            self.logger.info("🎥 No video tracks available")
            self._current_video_track_id = None
            return

        # Just pick the first available video track
        for track_id, (
            track_type,
            participant,
            forwarder,
        ) in self._active_video_tracks.items():
            # Only consider video tracks (camera or screenshare)
            if track_type not in (
                TrackType.TRACK_TYPE_VIDEO,
                TrackType.TRACK_TYPE_SCREEN_SHARE,
            ):
                continue

            track_type_name = TrackType.Name(track_type)
            self.logger.info(f"🎥 Switching to track: {track_type_name} ({track_id})")

            # Get the track and forwarder
            track = self.edge.add_track_subscriber(track_id)
            if track and forwarder and isinstance(self.llm, Realtime):
                # Send to Realtime provider
                await self.llm._watch_video_track(track, shared_forwarder=forwarder)
                self._current_video_track_id = track_id
                return
            else:
                self.logger.error(f"Failed to switch to track {track_id}")

        self.logger.warning("🎥 No suitable video tracks found")

    async def _process_track(self, track_id: str, track_type: int, participant):
        raw_forwarder = None
        processed_forwarder = None

        try:
            # we only process video tracks (camera video or screenshare)
            if track_type not in (
                TrackType.TRACK_TYPE_VIDEO,
                TrackType.TRACK_TYPE_SCREEN_SHARE,
            ):
                return

            # subscribe to the video track
            track = self.edge.add_track_subscriber(track_id)
            if not track:
                self.logger.error(f"Failed to subscribe to {track_id}")
                return

            # Wrap screenshare tracks to ensure even dimensions for H.264 encoding
            if track_type == TrackType.TRACK_TYPE_SCREEN_SHARE:

                class _EvenDimensionsTrack(VideoStreamTrack):
                    def __init__(self, src):
                        super().__init__()
                        self.src = src

                    async def recv(self):
                        return ensure_even_dimensions(await self.src.recv())

                track = _EvenDimensionsTrack(track)  # type: ignore[arg-type]

            # Create a SHARED VideoForwarder for the RAW incoming track
            # This prevents multiple recv() calls competing on the same track
            raw_forwarder = VideoForwarder(
                track,  # type: ignore[arg-type]
                max_buffer=30,
                fps=30,  # Max FPS for the producer (individual consumers can throttle down)
                name=f"raw_video_forwarder_{track_id}",
            )
            await raw_forwarder.start()
            self.logger.debug("🎥 Created raw VideoForwarder for track %s", track_id)

            # Track forwarders for cleanup
            self._video_forwarders.append(raw_forwarder)

            # Store track metadata
            self._active_video_tracks[track_id] = (
                track_type,
                participant,
                raw_forwarder,
            )

            # If Realtime provider supports video, switch to this new track
            track_type_name = TrackType.Name(track_type)

            if self.realtime_mode:
                if self._video_track:
                    # We have a video publisher (e.g., YOLO processor)
                    # Create a separate forwarder for the PROCESSED video track
                    self.logger.info(
                        "🎥 Forwarding PROCESSED video frames to Realtime provider"
                    )
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
                        await self.llm._watch_video_track(
                            self._video_track, shared_forwarder=processed_forwarder
                        )
                        self._current_video_track_id = track_id
                else:
                    # No video publisher, send raw frames - switch to this new track
                    self.logger.info(
                        f"🎥 Switching to {track_type_name} track: {track_id}"
                    )
                    if isinstance(self.llm, Realtime):
                        await self.llm._watch_video_track(
                            track, shared_forwarder=raw_forwarder
                        )
                        self._current_video_track_id = track_id

            has_image_processors = len(self.image_processors) > 0

            # video processors - pass the raw forwarder (they process incoming frames)
            for processor in self.video_processors:
                try:
                    await processor.process_video(
                        track, participant.user_id, shared_forwarder=raw_forwarder
                    )
                except Exception as e:
                    self.logger.error(
                        f"Error in video processor {type(processor).__name__}: {e}"
                    )

            # Use raw forwarder for image processors - only if there are image processors
            if not has_image_processors:
                # No image processors, just keep the connection alive
                self.logger.info(
                    "No image processors, video processing handled by video processors only"
                )
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

                        if has_image_processors:
                            img = video_frame.to_image()

                            for processor in self.image_processors:
                                try:
                                    await processor.process_image(
                                        img, participant.user_id
                                    )
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
            # Task was cancelled (e.g., track removed)
            # Clean up forwarders that were created for this track
            self.logger.debug(
                f"🎥 Cleaning up forwarders for cancelled track {track_id}"
            )

            # Stop and remove the raw forwarder if it was created
            if raw_forwarder is not None and hasattr(self, "_video_forwarders"):
                if raw_forwarder in self._video_forwarders:
                    try:
                        await raw_forwarder.stop()
                        self._video_forwarders.remove(raw_forwarder)
                    except Exception as e:
                        self.logger.error(f"Error stopping raw forwarder: {e}")

            # Stop and remove processed forwarder if it was created
            if processed_forwarder is not None and hasattr(self, "_video_forwarders"):
                if processed_forwarder in self._video_forwarders:
                    try:
                        await processed_forwarder.stop()
                        self._video_forwarders.remove(processed_forwarder)
                    except Exception as e:
                        self.logger.error(f"Error stopping processed forwarder: {e}")

            return

    async def _on_turn_event(self, event: TurnStartedEvent | TurnEndedEvent) -> None:
        """Handle turn detection events."""
        # In realtime mode, the LLM handles turn detection, interruption, and responses itself
        if self.realtime_mode:
            return

        if isinstance(event, TurnStartedEvent):
            # Interrupt TTS when user starts speaking (barge-in)
            if event.participant and event.participant.user_id != self.agent_user.id:
                if self.tts:
                    self.logger.info(
                        f"👉 Turn started - interrupting TTS for participant {event.participant.user_id}"
                    )
                    try:
                        await self.tts.stop_audio()
                    except Exception as e:
                        self.logger.error(f"Error stopping TTS: {e}")
                else:
                    participant_id = event.participant.user_id if event.participant else "unknown"
                    self.logger.info(
                        f"👉 Turn started - participant speaking {participant_id} : {event.confidence}"
                    )
            else:
                # Agent itself started speaking - this is normal
                participant_id = event.participant.user_id if event.participant else "unknown"
                self.logger.debug(
                    f"👉 Turn started - agent speaking {participant_id}"
                )
        elif isinstance(event, TurnEndedEvent):
            participant_id = event.participant.user_id if event.participant else "unknown"
            self.logger.info(
                f"👉 Turn ended - participant {participant_id} finished (confidence: {event.confidence})"
            )

            # When turn detection is enabled, trigger LLM response when user's turn ends
            # This is the signal that the user has finished speaking and expects a response
            if event.participant and event.participant.user_id != self.agent_user.id:
                # Get the accumulated transcript for this speaker
                transcript = self._pending_user_transcripts.get(
                    event.participant.user_id, ""
                )

                if transcript and transcript.strip():
                    self.logger.info(
                        f"🤖 Triggering LLM response after turn ended for {event.participant.user_id}"
                    )

                    # Create participant object if we have metadata
                    participant = None
                    if hasattr(event, "custom") and event.custom:
                        # Try to extract participant info from custom metadata
                        participant = event.custom.get("participant")

                    # Trigger LLM response with the complete transcript
                    if self.llm:
                        await self.simple_response(transcript, participant)

                    # Clear the pending transcript for this speaker
                    self._pending_user_transcripts[event.participant.user_id] = ""

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
        """Whether the agent should publish an outbound video track."""
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
                # Default to WebRTC-friendly format unless configured differently
                framerate = 48000
                stereo = True
                self._audio_track = self.edge.create_audio_track(
                    framerate=framerate, stereo=stereo
                )
                # Inform TTS of desired output format so it can resample accordingly
                if self.tts:
                    channels = 2 if stereo else 1
                    self.tts.set_output_format(
                        sample_rate=framerate,
                        channels=channels,
                    )

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
