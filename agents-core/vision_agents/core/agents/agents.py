import asyncio
import datetime
import logging
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    TypeGuard,
)
from uuid import uuid4

import getstream.models
from aiortc import VideoStreamTrack
from getstream.video.rtc import Call
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import TrackType
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.context import Token
from opentelemetry.trace import Tracer, set_span_in_context
from opentelemetry.trace.propagation import Context, Span

from ..edge import sfu_events
from ..edge.events import (
    AudioReceivedEvent,
    CallEndedEvent,
    TrackAddedEvent,
    TrackRemovedEvent,
)
from ..edge.types import OutputAudioTrack, Participant, PcmData, User
from ..events.manager import EventManager
from ..instructions import Instructions
from ..llm import events as llm_events
from ..llm.events import (
    LLMResponseChunkEvent,
    LLMResponseCompletedEvent,
    RealtimeAgentSpeechTranscriptionEvent,
    RealtimeAudioOutputEvent,
    RealtimeUserSpeechTranscriptionEvent,
)
from ..llm.llm import LLM, AudioLLM, VideoLLM
from ..llm.realtime import Realtime
from ..mcp import MCPBaseServer, MCPManager
from ..processors.base_processor import (
    AudioProcessor,
    AudioPublisher,
    Processor,
    VideoProcessor,
    VideoPublisher,
)
from ..profiling import Profiler
from ..stt.events import STTErrorEvent, STTPartialTranscriptEvent, STTTranscriptEvent
from ..stt.stt import STT
from ..tts.events import TTSAudioEvent
from ..tts.tts import TTS
from ..turn_detection import TurnDetector, TurnEndedEvent, TurnStartedEvent
from ..utils.audio_queue import AudioQueue
from ..utils.logging import (
    CallContextToken,
    clear_call_context,
    set_call_context,
)
from ..utils.utils import await_or_run, cancel_and_wait
from ..utils.video_forwarder import VideoForwarder
from ..utils.video_track import VideoFileTrack
from . import events
from .agent_types import AgentOptions, LLMTurn, TrackInfo, default_agent_options
from .conversation import Conversation
from .transcript_buffer import TranscriptBuffer

if TYPE_CHECKING:
    from vision_agents.plugins.getstream.stream_edge_transport import (
        StreamConnection,
        StreamEdge,
    )

logger = logging.getLogger(__name__)

tracer: Tracer = trace.get_tracer("agents")


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

    Dev guidelines
    - Small methods so its easy to subclass/change behaviour
    """

    options: AgentOptions

    def __init__(
        self,
        # edge network for video & audio
        edge: "StreamEdge",
        # llm, optionally with sts/realtime capabilities
        llm: LLM | AudioLLM | VideoLLM,
        # the agent's user info
        agent_user: User,
        # instructions
        instructions: str = "Keep your replies short and dont use special characters.",
        # setup stt, tts, and turn detection if not using a realtime llm
        stt: Optional[STT] = None,
        tts: Optional[TTS] = None,
        turn_detection: Optional[TurnDetector] = None,
        # for video gather data at an interval
        # - roboflow/ yolo typically run continuously
        # - often combined with API calls to fetch stats etc
        # - state from each processor is passed to the LLM
        processors: Optional[List[Processor]] = None,
        # MCP servers for external tool and resource access
        mcp_servers: Optional[List[MCPBaseServer]] = None,
        options: Optional[AgentOptions] = None,
        tracer: Tracer = trace.get_tracer("agents"),
        profiler: Optional[Profiler] = None,
    ):
        self._agent_user_initialized = False
        self.agent_user = agent_user
        if not self.agent_user.id:
            self.agent_user.id = f"agent-{uuid4()}"

        self._pending_turn: Optional[LLMTurn] = None
        self.call: Optional[Call] = None

        self._active_processed_track_id: Optional[str] = None
        self._active_source_track_id: Optional[str] = None
        if options is None:
            options = default_agent_options()
        else:
            options = default_agent_options().update(options)
        self.options = options

        # audio incoming is enqueued to self._incoming_audio_queue (eg. human audio)
        self._incoming_audio_queue: AudioQueue = AudioQueue(buffer_limit_ms=8000)

        self.instructions = Instructions(input_text=instructions)
        self.edge = edge

        # OpenTelemetry data
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
        self.processors: list[Processor] = processors or []
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
        self._pending_user_transcripts: Dict[str, TranscriptBuffer] = defaultdict(
            TranscriptBuffer
        )

        # Merge plugin events BEFORE subscribing to any events
        for plugin in [stt, tts, turn_detection, llm, edge, profiler]:
            if plugin and hasattr(plugin, "events"):
                self.logger.debug(f"Register events from plugin {plugin}")
                self.events.merge(plugin.events)

        self.llm._attach_agent(self)

        # Attach processors that need agent reference
        for processor in self.processors:
            processor.attach_agent(self)

        self.events.subscribe(self._on_agent_say)

        # Track metadata: track_id -> TrackInfo
        self._active_video_tracks: Dict[str, TrackInfo] = {}
        self._video_forwarders: List[VideoForwarder] = []
        self._connection: Optional[StreamConnection] = None

        # Optional local video track override for debugging.
        # This track will play instead of any incoming video track.
        self._video_track_override_path: Optional[str | Path] = None

        # the outgoing audio track
        self._audio_track: Optional[OutputAudioTrack] = None

        # the outgoing video track
        self._video_track: Optional[VideoStreamTrack] = None

        self._audio_consumer_task: Optional[asyncio.Task] = None

        # validation time
        self._validate_configuration()
        self._prepare_rtc()

        # start audio consumption loop
        self.setup_event_handling()

        self.events.send(events.AgentInitEvent())

        # An event to detect if the call was ended.
        # `None` means the call is ended, or it hasn't started yet.
        # It is set only after agent joins the call
        self._call_ended_event: Optional[asyncio.Event] = None
        self._joined_at: float = 0.0

        self._join_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._closed = False

    async def _finish_llm_turn(self):
        if self._pending_turn is None or self._pending_turn.response is None:
            raise ValueError(
                "Finish LLM turn should only be called after self._pending_turn is set"
            )
        turn = self._pending_turn
        self._pending_turn = None
        event = turn.response
        if self.tts and event and event.text and event.text.strip():
            sanitized_text = self._sanitize_text(event.text)
            await self.tts.send(sanitized_text)

    def setup_event_handling(self):
        """
        Agent event handling:

        - STT: AudioReceivedEvent -> STTTranscriptEvent -> TurnCompleted -> LLMResponseCompletedEvent -> TTSAudioEvent
        - Eager: AudioReceivedEvent -> STTTranscriptEvent -> EagerTurnCompleted -> LLMResponseCompletedEvent
            - > if TurnCompleted -> TTSAudioEvent
        - Realtime: Transcriptions

        Other events
        - Tracks for video added/removed
        - Error events

        """
        # listen to turn completed, started etc
        self.events.subscribe(self._on_turn_event)

        @self.llm.events.subscribe
        async def on_llm_response_send_to_tts(event: LLMResponseCompletedEvent):
            # turns started outside of the agent (instructions from code)
            if self._pending_turn is None:
                if self.tts and event.text and event.text.strip():
                    sanitized_text = self._sanitize_text(event.text)
                    await self.tts.send(sanitized_text)
            else:
                self._pending_turn.response = event
                if self._pending_turn.turn_finished:
                    await self._finish_llm_turn()
                else:
                    # we are in eager turn completion mode. wait for confirmation
                    self._pending_turn.response = event

        # write tts pcm to output track (this is the AI talking to us)
        @self.events.subscribe
        async def _on_tts_audio_write_to_output(event: TTSAudioEvent):
            if self._audio_track is not None:
                await self._audio_track.write(event.data)

        # listen to video tracks added/removed
        @self.edge.events.subscribe
        async def on_video_track_added(event: TrackAddedEvent | TrackRemovedEvent):
            if event.track_id is None or event.track_type is None or event.user is None:
                return
            if isinstance(event, TrackRemovedEvent):
                asyncio.create_task(
                    self._on_track_removed(event.track_id, event.track_type, event.user)
                )
            else:
                asyncio.create_task(
                    self._on_track_added(event.track_id, event.track_type, event.user)
                )

        # audio event for the user talking to the AI
        @self.edge.events.subscribe
        async def on_audio_received(event: AudioReceivedEvent):
            if event.pcm_data is None:
                return

            await self._incoming_audio_queue.put(event.pcm_data)

        @self.edge.events.subscribe
        async def on_call_ended(event: CallEndedEvent):
            if self._call_ended_event is not None:
                self._call_ended_event.set()

            await self.close()

        @self.events.subscribe
        async def on_stt_transcript_event_create_response(
            event: STTTranscriptEvent | STTPartialTranscriptEvent,
        ):
            if _is_audio_llm(self.llm):
                # There is no need to send the response to the LLM if it handles audio itself.
                return

            if isinstance(event, STTPartialTranscriptEvent):
                self.logger.info(f"🎤 [Transcript Partial]: {event.text}")
            else:
                self.logger.info(f"🎤 [Transcript Complete]: {event.text}")

            user_id = event.user_id()
            if user_id is None:
                self.logger.warning("STT transcript event missing user_id, skipping")
                return

            # With turn detection: accumulate transcripts and wait for TurnEndedEvent
            self._pending_user_transcripts[user_id].update(event)

            # if turn detection is disabled, treat the transcript event as an end of turn
            if not self.turn_detection_enabled and isinstance(
                event, STTTranscriptEvent
            ):
                self.events.send(
                    TurnEndedEvent(
                        participant=event.participant,
                    )
                )

        # TODO: chat event handling needs work

        # Error handling
        @self.events.subscribe
        async def on_error(event: STTErrorEvent):
            self.logger.error("stt error event %s", event)

        @self.events.subscribe
        async def on_stt_transcript_event_sync_conversation(event: STTTranscriptEvent):
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

            if user_id := event.user_id():
                with self.span("agent.on_realtime_user_speech_transcription"):
                    await self.conversation.upsert_message(
                        message_id=str(uuid.uuid4()),
                        role="user",
                        user_id=user_id,
                        content=event.text,
                        completed=True,
                        replace=True,
                        original=event,
                    )
            else:
                self.logger.info(
                    "RealtimeUserSpeechTranscriptionEvent event does not contain a user, skip sync to chat"
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

        @self.llm.events.subscribe
        async def on_llm_response_sync_conversation(event: LLMResponseCompletedEvent):
            self.logger.info(f"🤖 [LLM response]: {event.text}")

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

    async def simple_response(
        self, text: str, participant: Optional[Participant] = None
    ) -> None:
        """
        Overwrite simple_response if you want to change how the Agent class calls the LLM
        """
        self.logger.info('🤖 Asking LLM to reply to "%s"', text)
        with self.tracer.start_as_current_span("simple_response") as span:
            await self.llm.simple_response(
                text=text, processors=self.processors, participant=participant
            )
            span.set_attribute("text", text)

    async def simple_audio_response(
        self, pcm: PcmData, participant: Optional[Participant] = None
    ) -> None:
        """
        Makes it easy to subclass how the agent calls the LLM for processing audio
        """
        if _is_audio_llm(self.llm):
            await self.llm.simple_audio_response(pcm, participant)

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

    @asynccontextmanager
    async def join(
        self, call: Call, participant_wait_timeout: Optional[float] = 10.0
    ) -> AsyncIterator[None]:
        """
        Join the given call.

        The agent can join the call only once.
        Once the call is ended, the agent closes itself.

        Args:
            call: the call to join.
            participant_wait_timeout: timeout in seconds to wait for other participants to join before proceeding.
                 If `0`, do not wait at all. If `None`, wait forever.
                 Default - `10.0`.

        Returns:

        """
        if self._call_ended_event is not None:
            raise RuntimeError("Agent already joined the call")

        try:
            await self._join_lock.acquire()
            self._start_tracing(call)
            self.call = call
            self.conversation = None

            # Ensure all subsequent logs include the call context.
            self._set_call_logging_context(call.id)

            # run start on all subclasses
            await self._apply("start")

            await self.create_user()

            # Connect to MCP servers if manager is available
            if self.mcp_manager:
                with self.span("mcp_manager.connect_all"):
                    await self.mcp_manager.connect_all()

            # Ensure Realtime providers are ready before proceeding (they manage their own connection)
            self.logger.info(f"🤖 Agent joining call: {call.id}")
            if _is_realtime_llm(self.llm):
                await self.llm.connect()

            with self.span("edge.join"):
                self._connection = await self.edge.join(self, call)
            self.logger.info(f"🤖 Agent joined call: {call.id}")

            # Set up audio and video tracks together to avoid SDP issues
            audio_track = self._audio_track if self.publish_audio else None
            video_track = self._video_track if self.publish_video else None

            if audio_track or video_track:
                with self.span("edge.publish_tracks"):
                    await self.edge.publish_tracks(audio_track, video_track)

            # Setup chat and connect it to transcript events
            self.conversation = await self.edge.create_conversation(
                call, self.agent_user, self.instructions.full_reference
            )

            # Provide conversation to the LLM so it can access the chat history.
            self.llm.set_conversation(self.conversation)

            if participant_wait_timeout != 0:
                await self.wait_for_participant(timeout=participant_wait_timeout)

            # Start consuming audio from the call
            self._audio_consumer_task = asyncio.create_task(
                self._consume_incoming_audio()
            )
            self._call_ended_event = asyncio.Event()
            self._joined_at = time.time()
            yield
        finally:
            await self.close()
            self._end_tracing()
            self._join_lock.release()

    async def wait_for_participant(self, timeout: Optional[float] = None) -> None:
        """
        Wait for a participant other than the AI agent to join.

        Args:
            timeout: How long to wait for the participant to join in seconds.
            If `None`, wait forever.
            Default - `30.0`.
        """
        if self._connection is None:
            return

        self.logger.info("Waiting for other participants to join")

        try:
            await self._connection.wait_for_participant(timeout=timeout)
        except asyncio.TimeoutError:
            self.logger.info(
                f"No participants joined after {timeout}s timeout, proceeding."
            )

    def idle_for(self) -> float:
        """
        Return the idle time for this connection if there is no other participants except the agent itself.
        `0.0` means that connection is active.

        Returns:
            idle time for this connection or 0.0
        """
        if self._connection is None or not self._joined_at:
            # The call hasn't started yet.
            return 0.0

        # The connection is opened, but it's not idle, exit early.
        idle_since = self._connection.idle_since()
        if not idle_since:
            return 0.0

        # The RTC connection is established and it's idle.
        # Adjust the idle_since timestamp if the Agent was waiting for participants before actually
        # joining the call.
        idle_since_adjusted = max(idle_since, self._joined_at)
        return time.time() - idle_since_adjusted

    async def finish(self):
        """
        Wait for the call to end gracefully.
        If no connection is active, returns immediately.
        """
        if self._call_ended_event is None:
            # Exit immediately because the agent either left the call, or the call hasn't even started.
            return

        try:
            await self._call_ended_event.wait()
        except asyncio.CancelledError:
            # Close the agent even if the coroutine is canceled
            self.events.send(events.AgentFinishEvent())
            await self.close()
            raise

    @contextmanager
    def span(self, name: str) -> Iterator[Span]:
        with self.tracer.start_as_current_span(name, context=self._root_ctx) as span:
            yield span

    def _start_tracing(self, call: Call) -> None:
        self._root_span = self.tracer.start_span("join").__enter__()
        self._root_span.set_attribute("call_id", call.id)
        if self.agent_user.id:
            self._root_span.set_attribute("agent_id", self.agent_user.id)
        self._root_ctx = set_span_in_context(self._root_span)
        # Activate the root context globally so all subsequent spans are nested under it
        self._context_token = otel_context.attach(self._root_ctx)

    async def _apply(self, function_name: str, *args, **kwargs):
        subclasses = [self.llm, self.stt, self.tts, self.turn_detection, self.edge]
        subclasses.extend(self.processors)
        for subclass in subclasses:
            if (
                subclass is not None
                and getattr(subclass, function_name, None) is not None
            ):
                func = getattr(subclass, function_name)
                if func is not None:
                    try:
                        await await_or_run(func, *args, **kwargs)
                    except Exception as e:
                        self.logger.exception(
                            f"Error calling {function_name} on {subclass.__class__.__name__}: {e}"
                        )

    def _end_tracing(self):
        if self._root_span is not None:
            self._root_span.__exit__(None, None, None)
            self._root_span = None
            self._root_ctx = None

        # Detach the context token if it was set
        if self._context_token is not None:
            otel_context.detach(self._context_token)
            self._context_token = None

    @property
    def closed(self) -> bool:
        return self._closed

    async def close(self):
        """
        Clean up all connections and resources.

        Closes MCP connections, realtime output, active media tracks, processor
        tasks, the call connection, STT/TTS services, and stops turn detection.
        It is safe to call multiple times.
        """
        if self._close_lock.locked() or self._closed:
            return

        async with self._close_lock:
            # This is how to make sure the `_stop()` coroutine is definitely finished even if the outer
            # task is cancelled.
            # Run _stop() in a shielded task
            task = asyncio.create_task(self._close())
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                # The close() itself is cancelled, but the shielded task is still running because that's
                # how shield() works.
                # Wait until the shielded task finishes
                await task
                # Propagate cancellation upwards
                raise

    async def _close(self):
        # Set call_ended event again in case the agent is closed externally
        self.logger.info("🤖 Stopping the agent")
        if self._call_ended_event is not None:
            self._call_ended_event.set()

        # Stop audio consumer task
        if self._audio_consumer_task:
            await cancel_and_wait(self._audio_consumer_task)
            self._audio_consumer_task = None

        # run stop on all subclasses
        await self._apply("stop")
        # run close on all subclasses
        await self._apply("close")

        # Disconnect from MCP servers
        if self.mcp_manager:
            await self.mcp_manager.disconnect_all()

        # Stop all video forwarders
        for forwarder in self._video_forwarders:
            try:
                await forwarder.stop()
            except Exception as e:
                self.logger.error(f"Error stopping video forwarder: {e}")
        self._video_forwarders.clear()

        # Close RTC connection
        if self._connection:
            await self._connection.close()
        self._connection = None

        # Stop audio track
        if self._audio_track:
            self._audio_track.stop()
        self._audio_track = None

        # Stop video track
        if self._video_track:
            self._video_track.stop()
        self._video_track = None

        self._call_ended_event = None
        self._joined_at = 0.0
        self.clear_call_logging_context()
        self._closed = True
        self.logger.info("🤖 Agent stopped")

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
            await self.edge.create_user(self.agent_user)
            self._agent_user_initialized = True

        return None

    async def create_call(self, call_type: str, call_id: str) -> Call:
        """Shortcut for creating a call/room etc."""
        call = self.edge.client.video.call(call_type, call_id)
        await call.get_or_create(data={"created_by_id": self.agent_user.id})

        return call

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

                sanitized_text = self._sanitize_text(event.text)
                await self.tts.send(sanitized_text, user_metadata)

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

    def set_video_track_override_path(self, path: str):
        if not path or not self.publish_video:
            return

        self.logger.warning(
            f'🎥 The video will be played from "{path}" instead of the call'
        )
        # Store the local video track.
        self._video_track_override_path = path

    async def _consume_incoming_audio(self) -> None:
        """Consumer that continuously processes audio from the queue."""
        interval_seconds = 0.02  # 20ms target interval

        try:
            while self._call_ended_event and not self._call_ended_event.is_set():
                loop_start = time.perf_counter()
                try:
                    # Get audio data from queue with timeout to keep the loop running
                    pcm = await asyncio.wait_for(
                        self._incoming_audio_queue.get_duration(duration_ms=20),
                        timeout=1.0,
                    )

                    participant = pcm.participant

                    if (
                        participant
                        and getattr(participant, "user_id", None) != self.agent_user.id
                    ):
                        # first forward to processors
                        for processor in self.audio_processors:
                            if processor is None:
                                continue
                            await processor.process_audio(pcm)

                        # when in Realtime mode call the Realtime directly (non-blocking)
                        if _is_audio_llm(self.llm):
                            await self.simple_audio_response(pcm, participant)

                        # Process audio through STT
                        elif self.stt:
                            await self.stt.process_audio(pcm, participant)

                    if self.turn_detection is not None and participant is not None:
                        await self.turn_detection.process_audio(
                            pcm, participant, conversation=self.conversation
                        )

                except (asyncio.TimeoutError, asyncio.QueueEmpty):
                    # No audio data available, continue the loop
                    pass

                # Sleep for remaining time to maintain consistent interval
                elapsed = time.perf_counter() - loop_start
                sleep_time = interval_seconds - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            self.logger.info("🎵 Audio consumer task cancelled")
            raise
        except Exception as e:
            self.logger.error(f"❌ Error in audio consumer: {e}", exc_info=True)

    async def _track_to_video_processors(self, track: TrackInfo):
        """
        Send the track to the video processors
        """
        # video processors - pass the raw forwarder (they process incoming frames)
        for processor in self.video_processors:
            try:
                user_id = track.participant.user_id if track.participant else None
                await processor.process_video(
                    track.track, user_id, shared_forwarder=track.forwarder
                )
            except Exception as e:
                self.logger.error(
                    f"Error in video processor {type(processor).__name__}: {e}"
                )

    async def _on_track_removed(
        self, track_id: str, track_type: int, participant: Participant
    ):
        # We only process video tracks (camera video or screenshare)
        if track_type not in (
            TrackType.TRACK_TYPE_VIDEO,
            TrackType.TRACK_TYPE_SCREEN_SHARE,
        ):
            return
        track_type_name = (
            "SCREEN_SHARE"
            if track_type == TrackType.TRACK_TYPE_SCREEN_SHARE
            else "VIDEO"
        )
        self.logger.info(
            f"📺 Track removed: {track_type_name} from {participant.user_id}"
        )

        track = self._active_video_tracks.pop(track_id, None)
        if track is not None:
            await track.forwarder.stop()
            track.track.stop()
            await self._on_track_change(track_id)

    async def _on_track_change(self, track_id: str):
        # shared logic between track remove and added
        # Select a track. Prioritize screenshare over regular
        # This is the track without processing
        non_processed_tracks = [
            t for t in self._active_video_tracks.values() if not t.processor
        ]
        if not non_processed_tracks:
            return
        source_track = sorted(
            non_processed_tracks, key=lambda t: t.priority, reverse=True
        )[0]
        # assign the tracks that we last used so we can notify of changes...
        self._active_source_track_id = source_track.id

        await self._track_to_video_processors(source_track)

        processed_track = sorted(
            [t for t in self._active_video_tracks.values()],
            key=lambda t: t.priority,
            reverse=True,
        )[0]
        self._active_processed_track_id = processed_track.id

        # See if we have a processed track. If so forward that to LLM
        # TODO: this should run in a loop and handle multiple forwarders

        # If Realtime provider supports video, switch to this new track
        if _is_video_llm(self.llm):
            await self.llm.watch_video_track(
                processed_track.track, shared_forwarder=processed_track.forwarder
            )

    async def _on_track_added(
        self, track_id: str, track_type: int, participant: Participant
    ):
        # We only process video tracks (camera video or screenshare)
        if track_type not in (
            TrackType.TRACK_TYPE_VIDEO,
            TrackType.TRACK_TYPE_SCREEN_SHARE,
        ):
            return

        track_type_name = (
            "SCREEN_SHARE"
            if track_type == TrackType.TRACK_TYPE_SCREEN_SHARE
            else "VIDEO"
        )
        self.logger.info(
            f"📺 Track added: {track_type_name} from {participant.user_id}"
        )

        if self._video_track_override_path is not None:
            # If local video track is set, we override all other video tracks with it.
            # We override tracks instead of simply playing one in order to keep the same lifecycle within the call.
            # Otherwise, we'd have a video going on without anybody on the call.
            track = await self._get_video_track_override()
        else:
            # Subscribe to the video track, we watch all tracks by default
            track = self.edge.add_track_subscriber(track_id)
            if not track:
                self.logger.error(f"Failed to subscribe to {track_id}")
                return

        # Store track metadata
        forwarder = VideoForwarder(
            track,  # type: ignore[arg-type]
            max_buffer=30,
            fps=30,  # Max FPS for the producer (individual consumers can throttle down)
            name=f"video_forwarder_{track_id}_{track_type}",
        )
        self._active_video_tracks[track_id] = TrackInfo(
            id=track_id,
            type=track_type,
            processor="",
            track=track,
            participant=participant,
            priority=1 if track_type == TrackType.TRACK_TYPE_SCREEN_SHARE else 0,
            forwarder=forwarder,
        )

        await self._on_track_change(track_id)

    async def _on_turn_event(self, event: TurnStartedEvent | TurnEndedEvent) -> None:
        """Handle turn detection events."""
        # Skip the turn event handling if the model doesn't require TTS or SST audio itself.
        if _is_audio_llm(self.llm):
            return

        if isinstance(event, TurnStartedEvent):
            # Interrupt TTS when user starts speaking (barge-in)
            if event.participant and event.participant.user_id != self.agent_user.id:
                if self.tts:
                    self.logger.info(
                        f"👉 Turn started - interrupting TTS for participant {event.participant.user_id}"
                    )
                    await self.tts.stop_audio()
                else:
                    participant_id = (
                        event.participant.user_id if event.participant else "unknown"
                    )
                    self.logger.info(
                        "👉 Turn started - participant speaking %s : %.2f",
                        participant_id,
                        event.confidence,
                    )
                if self._audio_track is not None:
                    await self._audio_track.flush()
            else:
                # Agent itself started speaking - this is normal
                participant_id = (
                    event.participant.user_id if event.participant else "unknown"
                )
                self.logger.debug(f"👉 Turn started - agent speaking {participant_id}")
        elif isinstance(event, TurnEndedEvent):
            participant_id = (
                event.participant.user_id if event.participant else "unknown"
            )
            self.logger.info(
                "👉 Turn ended - participant %s finished (confidence: %.2f)",
                participant_id,
                event.confidence,
            )
            if not event.participant or event.participant.user_id == self.agent_user.id:
                # Exit early if the event is triggered by the model response.
                return

            # When turn detection is enabled, trigger LLM response when user's turn ends.
            # This is the signal that the user has finished speaking and expects a response
            buffer = self._pending_user_transcripts[event.participant.user_id]

            # when turn is completed, wait for the last transcriptions

            if not event.eager_end_of_turn:
                if self.stt:
                    await self.stt.clear()
                    # give the speech to text a moment to catch up
                    await asyncio.sleep(0.02)

            # get the transcript, and reset the buffer if it's not an eager turn
            transcript = buffer.text
            if not event.eager_end_of_turn:
                buffer.reset()

            if transcript.strip():
                # cancel the old task if the text changed in the meantime

                if (
                    self._pending_turn is not None
                    and self._pending_turn.input != transcript
                ):
                    logger.debug(
                        "Eager turn and completed turn didn't match. Cancelling in flight response. %s vs %s ",
                        self._pending_turn.input,
                        transcript,
                    )
                    if self._pending_turn.task:
                        self._pending_turn.task.cancel()

                # create a new LLM turn
                if self._pending_turn is None or self._pending_turn.input != transcript:
                    # Without turn detection: trigger LLM immediately on transcript completion
                    # This is the traditional STT -> LLM flow
                    llm_turn = LLMTurn(
                        input=transcript,
                        participant=event.participant,
                        started_at=datetime.datetime.now(),
                        turn_finished=not event.eager_end_of_turn,
                    )
                    self._pending_turn = llm_turn
                    task = asyncio.create_task(
                        self.simple_response(transcript, event.participant)
                    )
                    llm_turn.task = task
                elif self._pending_turn.input == transcript:
                    # same text as pending turn
                    is_finished = not event.eager_end_of_turn
                    now = datetime.datetime.now()
                    elapsed = now - self._pending_turn.started_at
                    logger.debug(
                        "Marking eager turn as completed. Eager turn detection saved %.2f",
                        elapsed.total_seconds() * 1000,
                    )

                    if is_finished:
                        self._pending_turn.turn_finished = True
                        if self._pending_turn.response is not None:
                            await self._finish_llm_turn()

    @property
    def turn_detection_enabled(self):
        # return true if either turn detection or stt provide turn detection capabilities
        return self.turn_detection is not None or (
            self.stt is not None and self.stt.turn_detection
        )

    @property
    def publish_audio(self) -> bool:
        """Whether the agent should publish an outbound audio track.

        Returns:
            True if TTS is configured, when in Realtime mode, or if there are audio publishers.
        """
        if self.tts is not None or _is_audio_llm(self.llm):
            return True
        # Also publish audio if there are audio publishers (e.g., HeyGen avatar)
        if self.audio_publishers:
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
        needs_video = len(self.video_processors) > 0 or _is_video_llm(self.llm)

        return needs_audio or needs_video

    @property
    def audio_processors(self) -> list[AudioProcessor]:
        """Get processors that can process audio.

        Returns:
            List of processors that implement `process_audio(pcm_data: PcmData)`.
        """
        return [p for p in self.processors if isinstance(p, AudioProcessor)]

    @property
    def video_processors(self) -> list[VideoProcessor]:
        """Get processors that can process video.

        Returns:
            List of processors that implement `process_video(track, participant_id, shared_forwarder)`.
        """
        return [p for p in self.processors if isinstance(p, VideoProcessor)]

    @property
    def video_publishers(self) -> list[VideoPublisher]:
        """Get processors capable of publishing a video track.

        Returns:
            List of processors that implement `publish_video_track()`.
        """
        return [p for p in self.processors if isinstance(p, VideoPublisher)]

    @property
    def audio_publishers(self) -> list[AudioPublisher]:
        """Get processors capable of publishing an audio track.

        Returns:
            List of processors that implement `publish_audio_track()`.
        """
        return [p for p in self.processors if isinstance(p, AudioPublisher)]

    def _validate_configuration(self):
        """Validate the agent configuration."""
        if _is_audio_llm(self.llm):
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
            has_audio_processing = bool(self.stt or self.tts or self.turn_detection)
            has_video_processing = bool(self.video_processors)

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

        if self.publish_audio:
            framerate = 48000
            stereo = True
            self._audio_track = self.edge.create_audio_track(
                framerate=framerate, stereo=stereo
            )

            @self.events.subscribe
            async def forward_audio(event: RealtimeAudioOutputEvent):
                if self._audio_track is not None:
                    await self._audio_track.write(event.data)

        # Set up video track if video publishers are available
        if self.publish_video:
            # Get the first video publisher to create the track
            video_publisher = self.video_publishers[0]
            # TODO: some lLms like moondream publish video
            self._video_track = video_publisher.publish_video_track()
            forwarder = VideoForwarder(
                self._video_track,  # type: ignore[arg-type]
                max_buffer=30,
                fps=30,  # Max FPS for the producer (individual consumers can throttle down)
                name=f"video_forwarder_{video_publisher.name}",
            )
            self._active_video_tracks[self._video_track.id] = TrackInfo(
                id=self._video_track.id,
                type=TrackType.TRACK_TYPE_VIDEO,
                processor=video_publisher.name,
                track=self._video_track,
                participant=None,
                priority=2,
                forwarder=forwarder,
            )

            self.logger.info("🎥 Video track initialized from video publisher")

    def _sanitize_text(self, text: str) -> str:
        """Remove markdown and special characters that don't speak well."""
        return text.replace("*", "").replace("#", "")

    async def _get_video_track_override(self) -> VideoFileTrack:
        """
        Create a video track override in async way if the path is set.

        Returns: `VideoFileTrack`
        """
        if not self._video_track_override_path:
            raise ValueError("video_track_override_path is not set")
        return await asyncio.to_thread(
            lambda p: VideoFileTrack(p), self._video_track_override_path
        )


def _is_audio_llm(llm: LLM | VideoLLM | AudioLLM) -> TypeGuard[AudioLLM]:
    return isinstance(llm, AudioLLM)


def _is_video_llm(llm: LLM | VideoLLM | AudioLLM) -> TypeGuard[VideoLLM]:
    return isinstance(llm, VideoLLM)


def _is_realtime_llm(llm: LLM | AudioLLM | VideoLLM | Realtime) -> TypeGuard[Realtime]:
    return isinstance(llm, Realtime)


class _AgentLoggerAdapter(logging.LoggerAdapter):
    """
    A logger adapter to include the agent_id to the logs
    """

    def process(self, msg: str, kwargs):
        if self.extra:
            return "[Agent: %s] | %s" % (self.extra["agent_id"], msg), kwargs
        return super(_AgentLoggerAdapter, self).process(msg, kwargs)
