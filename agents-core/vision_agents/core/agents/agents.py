import asyncio
import logging
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import (
    AsyncIterator,
    Iterator,
    Optional,
    TypeGuard,
)
from uuid import uuid4

from aiortc import VideoStreamTrack
from getstream.video.rtc import AudioStreamTrack, PcmData
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.context import Token
from opentelemetry.trace import Tracer, set_span_in_context
from opentelemetry.trace.propagation import Context, Span

from ..avatars import Avatar
from ..edge import Call, EdgeTransport
from ..edge.events import (
    AudioReceivedEvent,
    CallEndedEvent,
    TrackAddedEvent,
    TrackRemovedEvent,
)
from ..edge.types import Connection, Participant, TrackType, User
from ..events.manager import EventManager
from ..instructions import Instructions
from ..llm import events as llm_events
from ..llm.llm import LLM, AudioLLM, VideoLLM
from ..llm.realtime import Realtime
from ..mcp import MCPBaseServer, MCPManager
from ..observability import MetricsCollector
from ..observability.agent import AgentMetrics
from ..processors.base_processor import (
    AudioProcessor,
    AudioPublisher,
    Processor,
    VideoProcessor,
    VideoPublisher,
)
from ..profiling import Profiler
from ..stt.stt import STT
from ..tts.tts import TTS
from ..turn_detection import TurnDetector
from ..utils.audio_filter import AudioFilter, FirstSpeakerWinsFilter
from ..utils.audio_queue import AudioQueue
from ..utils.logging import (
    CallContextToken,
)
from ..base import Component
from ..utils.exceptions import log_exceptions
from ..utils.utils import cancel_and_wait
from ..utils.video_forwarder import VideoForwarder
from ..utils.video_track import VideoFileTrack
from . import events
from .agent_types import AgentOptions, TrackInfo, default_agent_options
from .conversation import Conversation, InMemoryConversation
from .inference import (
    AudioInputChunk,
    AudioInputStream,
    AudioOutputChunk,
    AudioOutputFlush,
    AudioOutputStream,
    InferenceFlow,
    RealtimeInferenceFlow,
    TranscribingInferenceFlow,
)
from .transcript import TranscriptStore

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
    * agent.simple_response("greet the user")
    * await agent.finish() // (wait for the call session to finish)
    * agent.close() // cleanup

    Note: Don't reuse the agent object. Create a new agent object each time.
    """

    options: AgentOptions

    def __init__(
        self,
        # edge network for video & audio
        edge: EdgeTransport,
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
        processors: Optional[list[Processor]] = None,
        # optional avatar plugin. When set, the avatar owns the agent's
        # outbound video/audio tracks and the inference flow's audio output
        # is routed through the avatar provider for lipsync.
        avatar: Optional[Avatar] = None,
        # MCP servers for external tool and resource access
        mcp_servers: Optional[list[MCPBaseServer]] = None,
        options: Optional[AgentOptions] = None,
        tracer: Tracer = trace.get_tracer("agents"),
        profiler: Optional[Profiler] = None,
        # Metrics broadcasting to call participants
        broadcast_metrics: bool = False,
        broadcast_metrics_interval: float = 5.0,
        # Audio filter to process audio from multiple speakers
        multi_speaker_filter: Optional[AudioFilter] = None,
    ):
        """Initialize the Agent.

        Args:
            edge: Edge transport for video and audio connectivity.
            llm: LLM, optionally with audio/video/realtime capabilities.
            agent_user: The agent's user identity.
            instructions: System instructions for the LLM. Supports `@file.md` references.
            stt: Speech-to-text service. Not needed when using a realtime LLM.
            tts: Text-to-speech service. Not needed when using a realtime LLM.
            turn_detection: Turn detector for managing conversational turns.
                Not needed when using a realtime LLM.
            processors: Processors that run alongside the agent (e.g. video analysis,
                data fetching). Their state is passed to the LLM. Audio and video
                frames are dispatched to processors in list order; their
                lifecycle hooks (``start`` / ``close``) run concurrently, so
                processors must not depend on one another's startup or shutdown.
            avatar: Optional avatar plugin. When set, the avatar owns the
                agent's outbound video/audio tracks and the agent's
                audio output is routed through the avatar for lip-sync.
            mcp_servers: MCP servers for external tool and resource access.
            options: Agent configuration options. Merged with defaults when provided.
            tracer: OpenTelemetry tracer for distributed tracing.
            profiler: Optional profiler for performance monitoring.
            broadcast_metrics: Whether to periodically broadcast agent metrics
                to call participants as custom events.
            broadcast_metrics_interval: Interval in seconds between metric broadcasts.
            multi_speaker_filter: Audio filter for handling overlapping speech from
                multiple participants.
                Takes effect only more than one participant is present.
                Defaults to `FirstSpeakerWinsFilter`, which uses VAD to lock onto
                the first participant who starts speaking and drops audio from
                everyone else until the active speaker's turn ends, or they go
                silent.

        """
        self._agent_user_initialized = False
        self.agent_user = agent_user
        if self.agent_user.id:
            self._agent_user_id = self.agent_user.id
        else:
            self._agent_user_id = f"agent-{uuid4()}"
            self.agent_user.id = self._agent_user_id

        self._id = str(uuid4())
        self.call: Optional[Call] = None

        self._active_processed_track_id: Optional[str] = None
        self._active_source_track_id: Optional[str] = None
        if options is None:
            options = default_agent_options()
        else:
            options = default_agent_options().update(options)
        self.options = options

        # Per-participant audio queues keyed by participant.id
        self._participant_queues: dict[str, tuple[Participant, AudioQueue]] = {}
        self._audio_buffer_limit_ms = 8000

        # Built-in first-speaker-wins filter for multi-participant calls
        self._multi_speaker_filter: AudioFilter = (
            multi_speaker_filter or FirstSpeakerWinsFilter()
        )

        self.instructions = Instructions(input_text=instructions)
        self.edge = edge

        # OpenTelemetry data
        self.tracer = tracer
        self._root_span: Optional[Span] = None
        self._root_ctx: Optional[Context] = None

        self.logger = _AgentLoggerAdapter(logger, {"agent_id": self.agent_user.id})

        self.events = EventManager()
        self.events.register_events_from_module(events)
        self.events.register_events_from_module(llm_events)

        self.llm = llm
        self.stt = stt
        self.tts = tts
        self.turn_detection = turn_detection
        self.processors: list[Processor] = processors or []
        self.avatar = avatar
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
        # and chat integration
        self.transcripts = TranscriptStore(agent_user_id=self._agent_user_id)

        self._collector = MetricsCollector()
        # Merge plugin metric collectors so plugin on_*() calls forward to the agent root
        for plugin in [stt, tts, turn_detection, llm, avatar, *self.processors]:
            if plugin is not None:
                self._collector.merge(plugin.metrics)

        # Merge plugin events BEFORE subscribing to any events
        for component in [stt, tts, turn_detection, llm, avatar, edge, profiler]:
            if component is not None:
                self.logger.debug(f"Register events from plugin {component}")
                self.events.merge(component.events)

        self.llm._attach_agent(self)

        # Attach processors that need agent reference
        for processor in self.processors:
            processor.attach_agent(self)

        # Track metadata: track_id -> TrackInfo
        self._active_video_tracks: dict[str, TrackInfo] = {}
        self._video_forwarders: list[VideoForwarder] = []
        self._connection: Optional[Connection] = None

        # Optional local video track override for debugging.
        # This track will play instead of any incoming video track.
        self._video_track_override_path: Optional[str | Path] = None

        # the outgoing audio track
        self._audio_track: Optional[AudioStreamTrack] = None

        # the outgoing video track
        self._video_track: Optional[VideoStreamTrack] = None

        self._audio_consumer_task: Optional[asyncio.Task] = None
        self._audio_producer_task: Optional[asyncio.Task] = None
        self._metrics_broadcast_task: Optional[asyncio.Task] = None

        # Metrics broadcasting settings
        self._broadcast_metrics = broadcast_metrics
        self._broadcast_metrics_interval = broadcast_metrics_interval

        # Validate the Agent is configured correctly
        self._validate_configuration()

        self._setup_media_tracks()

        self._subscribe_to_edge_events()

        # An event to detect if the call was ended.
        # `None` means the call is ended, or it hasn't started yet.
        # It is set only after agent joins the call
        self._call_ended_event: Optional[asyncio.Event] = None
        self._joined_at: float = 0.0

        self._join_lock = asyncio.Lock()
        self._authenticate_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._closed = False

        self._audio_input_stream = AudioInputStream()
        self._audio_output_stream = AudioOutputStream()
        if self.avatar is not None:
            # The agent's audio output is the avatar's input
            self.avatar.attach_audio_input(self._audio_output_stream)
        self._flow = self.resolve_inference_flow()

    @property
    def id(self) -> str:
        return self._id

    def _subscribe_to_edge_events(self):
        """
        Agent event handling:

        - Tracks for video added/removed
        - Error events

        """

        @self.edge.events.subscribe
        async def on_video_track_added(event: TrackAddedEvent | TrackRemovedEvent):
            # listen to video tracks added/removed
            if (
                event.track_id is None
                or event.track_type is None
                or event.participant is None
            ):
                return
            if isinstance(event, TrackRemovedEvent):
                asyncio.create_task(
                    self._on_track_removed(
                        event.track_id, event.track_type, event.participant
                    )
                )
            else:
                asyncio.create_task(
                    self._on_track_added(
                        event.track_id, event.track_type, event.participant
                    )
                )

        # audio event for the user talking to the AI
        @self.edge.events.subscribe
        async def on_audio_received(event: AudioReceivedEvent):
            if event.pcm_data is None or event.participant is None:
                return
            pcm = event.pcm_data
            participant = event.participant
            existing = self._participant_queues.get(participant.id)
            if existing is not None:
                _, queue = existing
                await queue.put(pcm)
            else:
                queue = AudioQueue(buffer_limit_ms=self._audio_buffer_limit_ms)
                await queue.put(pcm)
                self._participant_queues[participant.id] = (participant, queue)

        @self.edge.events.subscribe
        async def on_call_ended(_: CallEndedEvent):
            if self._call_ended_event is not None:
                self._call_ended_event.set()

            await self.close()

    async def simple_response(
        self,
        text: str,
        participant: Optional[Participant] = None,
        interrupt: bool = True,
    ) -> None:
        """Ask the LLM to reply to an injected instruction.

        The request is routed through the agent's inference flow so it shares
        the same LLM/TTS/audio pipeline as speech-driven turns.

        Args:
            text: Instruction or message to inject.
            participant: Participant the injected turn is attributed to.
                Defaults to the agent itself when not supplied.
            interrupt: If True (default), preempt any in-flight LLM turn. If
                False, drop silently when a turn is already in flight.
        """
        with self.tracer.start_as_current_span("agent.simple_response"):
            if participant is None:
                participant = Participant(
                    original=self.agent_user,
                    user_id=self._agent_user_id,
                    id=self._agent_user_id,
                )
            await self._flow.simple_response(text, participant, interrupt=interrupt)

    async def say(self, text: str, interrupt: bool = False) -> None:
        """Speak ``text`` directly through TTS, bypassing the LLM.

        Args:
            text: The utterance to speak.
            interrupt: If True, preempt any in-flight turn and clear the TTS
                pipeline first. If False (default), queue behind ongoing speech.
        """
        self.logger.info('🔊 Agent say: "%s"', text)
        with self.tracer.start_as_current_span("agent.say"):
            await self._flow.say(text, interrupt=interrupt)

        if self.conversation is not None:
            await self.conversation.upsert_message(
                role="assistant",
                user_id=self._agent_user_id,
                content=text,
                completed=True,
            )

    def resolve_inference_flow(self) -> InferenceFlow:
        """
        Picks InferenceFlow for the Agent based on provided plugins.
        Default behavior:
        - If provided LLM is `Realtime`, picks `RealtimeInferenceFlow`
        - Otherwise, picks `TranscribingInferenceFlow`

        This method can be overridden by subclasses to use custom
        InferenceFlow implementations.

        Returns:
            InferenceFlow instance

        """
        if _is_realtime_llm(self.llm):
            return RealtimeInferenceFlow(
                audio_input=self._audio_input_stream,
                audio_output=self._audio_output_stream,
                llm=self.llm,
                conversation=self.conversation
                or InMemoryConversation(
                    messages=[], instructions=self.instructions.full_reference
                ),
                transcripts=self.transcripts,
                agent_user_id=self._agent_user_id,
                events=self.events,
            )
        else:
            return TranscribingInferenceFlow(
                audio_input=self._audio_input_stream,
                audio_output=self._audio_output_stream,
                stt=self.stt,
                llm=self.llm,
                tts=self.tts,
                turn_detector=self.turn_detection,
                conversation=self.conversation
                or InMemoryConversation(
                    messages=[], instructions=self.instructions.full_reference
                ),
                transcripts=self.transcripts,
                agent_user_id=self._agent_user_id,
                events=self.events,
            )

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

            await self._start_components()

            # Connect to MCP servers if manager is available
            if self.mcp_manager:
                with self.span("mcp_manager.connect_all"):
                    await self.mcp_manager.connect_all()

            # Ensure Realtime providers are ready before proceeding (they manage their own connection)
            self.logger.info(f"🤖 Agent joining call: {call.id}")
            if _is_realtime_llm(self.llm):
                await self.llm.connect()

            # Authenticate an agent before calling edge.join()
            await self.authenticate()
            with self.span("edge.join"):
                self._connection = await self.edge.join(self, call)
            self.logger.info(f"🤖 Agent joined call: {call.id}")
            self.events.send(events.AgentJoinedCallEvent(call=call))

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

            # Provide conversation to other components.
            self._flow.set_conversation(self.conversation)
            self.llm.set_conversation(self.conversation)

            if participant_wait_timeout != 0:
                await self.wait_for_participant(timeout=participant_wait_timeout)

            await self._flow.start()

            # Start consuming audio from the call
            self._audio_consumer_task = asyncio.create_task(
                self._consume_incoming_audio()
            )
            if self.publish_audio:
                self._audio_producer_task = asyncio.create_task(
                    self._produce_audio_output()
                )

            # Start metrics broadcast if enabled
            if self._broadcast_metrics:
                self._metrics_broadcast_task = asyncio.create_task(
                    self._metrics_broadcast_loop()
                )

            self._call_ended_event = asyncio.Event()
            self._joined_at = time.time()
            yield
        except Exception as exc:
            if self._closing or self._closed:
                # Only log exceptions if the agent is already closing
                # (e.g., when the call ended before the agent fully joined).
                logger.warning(
                    f"Failed to join the call because the agent is closing or already closed: {exc}"
                )
                # Yield to let the context manager proceed
                yield
            else:
                raise
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

    def on_call_for(self) -> float:
        """
        Return the number of seconds for how long the agent has been on the call.
        Returns 0.0 if the agent has not joined a call yet.

        Returns:
            Duration in seconds since the agent joined the call, or 0.0 if not on a call.
        """
        if not self._joined_at:
            return 0.0
        return time.time() - self._joined_at

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

    def _get_components(self) -> tuple[Component, ...]:
        """All agent-owned components that implement the Lifecycle protocol."""
        return tuple(
            c
            for c in (
                self.llm,
                self.stt,
                self.tts,
                self.turn_detection,
                self.edge,
                self.avatar,
                *self.processors,
            )
            if c is not None
        )

    async def _start_components(self) -> None:
        """Start all components concurrently; abort the agent if any fails.

        Errors are logged with the failing component's name and re-raised; in
        flight siblings are cancelled and awaited so we don't leak orphan
        network connects.
        """

        async def _safe_start(component: Component) -> None:
            with log_exceptions(
                self.logger,
                f"Error starting {type(component).__name__}",
                reraise=True,
            ):
                await component.start()

        tasks = [asyncio.create_task(_safe_start(c)) for c in self._get_components()]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def _stop_components(self) -> None:
        """Close all components concurrently. Errors are logged but swallowed."""

        async def _safe_close(component: Component) -> None:
            with log_exceptions(
                self.logger,
                f"Error closing {type(component).__name__}",
            ):
                await component.close()

        await asyncio.gather(*(_safe_close(c) for c in self._get_components()))

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
        if self.call is not None:
            self.events.send(events.AgentLeftCallEvent(call=self.call))
        if self._call_ended_event is not None:
            self._call_ended_event.set()

        # Stop audio consumer task
        if self._audio_consumer_task:
            await cancel_and_wait(self._audio_consumer_task)
            self._audio_consumer_task = None

        if self._audio_producer_task:
            await cancel_and_wait(self._audio_producer_task)
            self._audio_producer_task = None

        # Stop the inference flow
        await self._flow.stop()
        self._audio_input_stream.close()
        self._audio_output_stream.close()

        # Drain any background conversation persistence so we don't drop
        # in-flight Stream Chat writes (e.g. the final assistant message).
        if self.conversation is not None:
            await self.conversation.wait_for_pending_syncs()

        # Stop metrics broadcast task
        if self._metrics_broadcast_task:
            await cancel_and_wait(self._metrics_broadcast_task)
            self._metrics_broadcast_task = None

        await self._stop_components()

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
        await self.events.shutdown()
        self._closed = True
        self.logger.info("🤖 Agent stopped")

    @property
    def _closing(self):
        return self._close_lock.locked()

    async def authenticate(self) -> None:
        """Authenticate the agent user with the edge provider.

        Idempotent — safe to call multiple times.
        """
        async with self._authenticate_lock:
            if self._agent_user_initialized:
                return None

            with self.span("edge.authenticate"):
                await self.edge.authenticate(self.agent_user)
                self._agent_user_initialized = True

        return None

    async def create_call(self, call_type: str, call_id: str) -> Call:
        """Create a call in the edge provider.

        Automatically authenticates if not already done.
        """
        await self.authenticate()
        call = await self.edge.create_call(
            call_id=call_id, agent_user_id=self.agent_user.id, call_type=call_type
        )
        return call

    def set_video_track_override_path(self, path: str):
        if not path or not self.publish_video:
            return

        self.logger.warning(
            f'🎥 The video will be played from "{path}" instead of the call'
        )
        # Store the local video track.
        self._video_track_override_path = path

    async def _poll_audio_queues(self) -> AsyncIterator[tuple[Participant, PcmData]]:
        """Yield 20 ms chunks, draining the whole backlog per queue so the
        consumer can catch up after a stall."""
        # Make a copy before iterating because the track maybe get unpublished
        queues = self._participant_queues.copy()
        for participant, queue in queues.values():
            while not queue.empty():
                try:
                    pcm = await asyncio.wait_for(
                        queue.get_duration(duration_ms=20), timeout=0.001
                    )
                    yield participant, pcm
                except (TimeoutError, asyncio.QueueEmpty):
                    break

    async def _consume_incoming_audio(self) -> None:
        """Consumer that continuously processes audio from per-participant queues."""
        interval_seconds = 0.02  # 20ms target interval

        # Store audio processors in the variable to avoid calling
        # the property all the time
        audio_processors = self.audio_processors

        try:
            while self._call_ended_event and not self._call_ended_event.is_set():
                loop_start = time.perf_counter()

                async for participant, pcm in self._poll_audio_queues():
                    if participant.user_id != self.agent_user.id:
                        # Pass audio through the filter
                        # if multiple participants are on the call
                        if len(self._participant_queues) > 1:
                            pcm = await self._multi_speaker_filter.process_audio(
                                pcm, participant
                            )
                            if pcm is None:
                                continue

                        await self._audio_input_stream.send(
                            AudioInputChunk(data=pcm, participant=participant)
                        )

                        # Pass PCM through audio processors
                        for processor in audio_processors:
                            await processor.process_audio(pcm)

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

    async def _produce_audio_output(self):
        if self._audio_track is None:
            return

        # When `avatar` is provided, we use its audio output instead of the
        # default one.
        stream = (
            self.avatar.audio_output()
            if self.avatar is not None
            else self._audio_output_stream
        )
        try:
            async for audio_output in stream:
                match audio_output:
                    case AudioOutputChunk(data=data) if data is not None:
                        await self._audio_track.write(data)
                    case AudioOutputFlush():
                        await self._audio_track.flush()

        except asyncio.CancelledError:
            self.logger.info("🎵 Audio producer task cancelled")
            raise
        except Exception:
            self.logger.exception("❌ Error in audio producer task")

    async def _metrics_broadcast_loop(self):
        """Background task that periodically broadcasts metrics to call participants."""
        try:
            self.logger.info(
                f"📊 Starting metrics broadcast (interval: {self._broadcast_metrics_interval}s)"
            )
            while True:
                await asyncio.sleep(self._broadcast_metrics_interval)
                try:
                    await self.send_metrics_event()
                except RuntimeError:
                    # Not connected to a call, skip this iteration
                    pass
                except Exception as e:
                    self.logger.warning(f"Failed to broadcast metrics: {e}")
        except asyncio.CancelledError:
            self.logger.info("📊 Metrics broadcast task cancelled")
            raise

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
        self, track_id: str, track_type: TrackType, participant: Participant
    ):
        if track_type == TrackType.AUDIO:
            self._participant_queues.pop(participant.id, None)
            self._multi_speaker_filter.clear(participant)
            return

        # We only process video tracks (camera video or screenshare)
        if track_type not in (
            TrackType.VIDEO,
            TrackType.SCREEN_SHARE,
        ):
            return

        self.logger.info(
            f"📺 Track removed: {track_type.name} from {participant.user_id}"
        )

        track = self._active_video_tracks.pop(track_id, None)
        if track is not None:
            await track.forwarder.stop()
            await self._on_track_change(track_id)

    async def _on_track_change(self, _: str):
        # shared logic between track remove and added
        # Select a track. Prioritize screenshare over regular
        # This is the track without processing
        non_processed_tracks = [
            t for t in self._active_video_tracks.values() if not t.processor
        ]
        if not non_processed_tracks:
            # No active video tracks left, stop sending video to the LLM and processors
            if _is_video_llm(self.llm):
                await self.llm.stop_watching_video_track()
            for processor in self.video_processors:
                await processor.stop_processing()
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
        self, track_id: str, track_type: TrackType, participant: Participant
    ):
        # We only process video tracks (camera video or screenshare)
        if track_type not in (
            TrackType.VIDEO,
            TrackType.SCREEN_SHARE,
        ):
            return

        if not self._needs_video():
            return

        self.logger.info(
            f"📺 Track added: {track_type.name} from {participant.user_id}"
        )

        track: VideoStreamTrack | None
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
            priority=1 if track_type == TrackType.SCREEN_SHARE else 0,
            forwarder=forwarder,
        )

        await self._on_track_change(track_id)

    @property
    def audio_track(self) -> Optional[AudioStreamTrack]:
        """The outgoing audio track published to the call."""
        return self._audio_track

    @property
    def publish_audio(self) -> bool:
        """Whether the agent should publish an outbound audio track.

        Returns:
            True if TTS is configured, when in Realtime mode, or if there are audio publishers.
        """
        if self.tts is not None or _is_audio_llm(self.llm):
            return True

        if self.audio_publishers:
            return True
        return False

    @property
    def publish_video(self) -> bool:
        """Whether the agent should publish an outbound video track."""

        # Avatars always publish video track
        if self.avatar is not None:
            return True
        return len(self.video_publishers) > 0

    def _needs_video(self) -> bool:
        return len(self.video_processors) > 0 or _is_video_llm(self.llm)

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
            if self.stt or self.tts or self.turn_detection:
                self.logger.warning(
                    "Realtime mode detected: STT, TTS and Turn Detection services will be disabled. "
                    "The Realtime model handles speech-to-text, text-to-speech and turn detection internally."
                )
                self.stt = None
                self.tts = None
                self.turn_detection = None
        else:
            if self.turn_detection and self.stt and self.stt.turn_detection:
                self.logger.warning(
                    "STT already provides turn detection; ignoring the TurnDetector plugin."
                )
                self.turn_detection = None

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

    def _setup_media_tracks(self):
        if self.publish_audio:
            if self.audio_publishers:
                self._audio_track = self.audio_publishers[0].publish_audio_track()
            else:
                self._audio_track = self.edge.create_audio_track()

        if not self.publish_video:
            return

        # Avatar owns the outbound video track. Its track is NOT placed in
        # `_active_video_tracks` — that dict is for inputs the LLM and
        # processors might watch.
        if self.avatar is not None:
            self._video_track = self.avatar.video_output()
            return

        video_publisher = self.video_publishers[0]
        self._video_track = video_publisher.publish_video_track()
        forwarder = VideoForwarder(
            self._video_track,  # type: ignore[arg-type]
            max_buffer=30,
            fps=30,
            # Max FPS for the producer (individual consumers can throttle down)
            name=f"video_forwarder_{video_publisher.name}",
        )
        self._active_video_tracks[self._video_track.id] = TrackInfo(
            id=self._video_track.id,
            type=TrackType.VIDEO.value,
            processor=video_publisher.name,
            track=self._video_track,
            participant=None,
            priority=2,
            forwarder=forwarder,
        )

        self.logger.info("🎥 Video track initialized from video publisher")

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

    @property
    def metrics(self) -> AgentMetrics:
        return self._collector.agent_metrics

    async def send_custom_event(self, data: dict) -> None:
        """Send a custom event to all participants watching the call.

        Custom events are delivered to clients via `call.on("custom", callback)`.
        Use this to send real-time data to your app's UI, such as metrics,
        status updates, or any custom application data.

        Args:
            data: Custom event payload (must be JSON-serializable, max 5KB).

        Raises:
            RuntimeError: If the agent is not connected to a call.

        Example:
            await agent.send_custom_event({
                "type": "status_update",
                "status": "processing",
                "progress": 0.5
            })
        """
        await self.edge.send_custom_event(data)

    async def send_metrics_event(
        self, event_type: str = "agent_metrics", fields: list[str] | None = None
    ) -> None:
        """Send current agent metrics as a custom event.

        This is a convenience method that packages the agent's metrics
        and sends them as a custom event to all participants.

        Args:
            event_type: The type identifier for the event (default: "agent_metrics").
            fields: Optional list of specific metric fields to include.
                   If None, includes all metrics.

        Example:
            # Send all metrics
            await agent.send_metrics_event()

            # Send only LLM-related metrics
            await agent.send_metrics_event(fields=[
                "llm_latency_ms__avg",
                "llm_input_tokens__total",
                "llm_output_tokens__total"
            ])
        """
        metrics_data = self._collector.agent_metrics.to_dict(fields or ())
        await self.send_custom_event(
            {
                "type": event_type,
                "metrics": metrics_data,
            }
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
