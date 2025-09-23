import asyncio
import logging
import time
from typing import Optional, List, Any, Union
from uuid import uuid4

import aiortc
from aiortc import VideoStreamTrack
from aiortc.contrib.media import MediaRelay

from ..edge.types import Participant, PcmData, Connection, TrackType, User
from ..llm.events import RealtimePartialTranscriptEvent
from ..edge.events import AudioReceivedEvent, TrackAddedEvent, CallEndedEvent
from ..llm.events import StandardizedTextDeltaEvent
from ..tts.tts import TTS
from ..stt.stt import STT
from ..vad import VAD
from ..llm.events import RealtimeTranscriptEvent, LLMResponseEvent
from ..stt.events import STTTranscriptEvent, STTPartialTranscriptEvent
from ..vad.events import VADAudioEvent
from getstream.video.rtc import Call
from ..mcp import MCPBaseServer
from ..mcp.tool_converter import MCPToolConverter


from .conversation import StreamHandle, Message, Conversation
from ..events.manager import EventManager
from ..llm.llm import LLM
from ..llm.realtime import Realtime
from ..processors.base_processor import filter_processors, ProcessorType, BaseProcessor
from . import events
from ..turn_detection import TurnEvent, TurnEventData, BaseTurnDetector
from typing import TYPE_CHECKING, Dict

import getstream.models

if TYPE_CHECKING:
    from stream_agents.plugins.getstream.stream_edge_transport import StreamEdge
    from .agent_session import AgentSessionContextManager

logger = logging.getLogger(__name__)


def _log_task_exception(task: asyncio.Task):
    try:
        task.result()
    except Exception as e:
        logger.exception("Error in background task")

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
        edge: "StreamEdge",
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

        # we sync the user talking and the agent responses to the conversation
        # because we want to support streaming responses and can have delta updates for both
        # user and agent we keep an handle for both
        self.conversation: Optional[Conversation] = None
        self._user_conversation_handle: Optional[StreamHandle] = None
        self._agent_conversation_handle: Optional[StreamHandle] = None

        # Merge plugin events BEFORE subscribing to any events
        for plugin in [stt, tts, turn_detection, vad, llm]:
            if plugin and hasattr(plugin, "events"):
                self.logger.info(f"Registered plugin {plugin}")
                self.events.merge(plugin.events)

        if self.llm is not None:
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
        self._connection: Optional[Connection] = None
        self._audio_track: Optional[aiortc.AudioStreamTrack] = None
        self._video_track: Optional[VideoStreamTrack] = None
        self._sts_connection = None
        self._pc_track_handler_attached: bool = False

        # validation time
        self._validate_configuration()

        # Initialize track management attributes
        self._active_tracks: Dict[str, Any] = {}  # Track active video/audio tracks
        self._last_health_check: float = 0.0  # Last health check timestamp
        self._track_health_check_interval: float = (
            1.0  # Health check interval in seconds
        )

        self._prepare_rtc()
        self._setup_stt()
        self._setup_turn_detection()

    def _truncate_for_logging(self, obj, max_length=200):
        """Truncate object string representation for logging to prevent spam."""
        obj_str = str(obj)
        if len(obj_str) > max_length:
            obj_str = obj_str[:max_length] + "... (truncated)"
        return obj_str

    def _setup_mcp_servers(self):
        """Set up MCP servers if any are configured."""
        if self.mcp_servers:
            self.logger.info(f"🔌 Setting up {len(self.mcp_servers)} MCP server(s)")
            for i, server in enumerate(self.mcp_servers):
                self.logger.info(f"  {i + 1}. {server.__class__.__name__}")
        else:
            self.logger.debug("No MCP servers configured")

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

        # Clean up active tracks
        self.logger.info(f"🎥VDP: Cleaning up {len(self._active_tracks)} active tracks")
        for track_id in list(self._active_tracks.keys()):
            self._remove_track(track_id)

        # Clean up track processing tasks
        if hasattr(self, "_track_tasks"):
            self.logger.info(
                f"🎥VDP: Canceling {len(self._track_tasks)} track processing tasks"
            )
            for task in self._track_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass  # Expected when canceling
                    except Exception as e:
                        self.logger.debug(f"🎥VDP: Error during task cancellation: {e}")
            self._track_tasks.clear()

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

    def subscribe(self, function):
        """Subscribe to event"""
        return self.events.subscribe(function)

    async def join(self, call: Call) -> "AgentSessionContextManager":
        self.call = call
        self.conversation = None

        # Connect to MCP servers
        await self._connect_mcp_servers()

        # Only set up chat if we have LLM (for conversation capabilities)
        if self.llm:
            # ask the edge to start the chat
            self.conversation = self.edge.create_conversation(
                call, self.agent_user, self.instructions
            )

        # when using STS, we sync conversation using transcripts otherwise we fallback to ST (if available)
        if self.sts_mode:
            self.events.subscribe(self._on_transcript)
            self.events.subscribe(self._on_partial_transcript)
        elif self.stt:
            self.events.subscribe(self._on_transcript)
            self.events.subscribe(self._on_partial_transcript)

        """Join a Stream video call."""
        if self._is_running:
            raise RuntimeError("Agent is already running")

        self.logger.info(f"🤖 Agent joining call: {call.id}")

        # Ensure Realtime providers are ready before proceeding (they manage their own connection)
        if self.sts_mode and isinstance(self.llm, Realtime):
            await self.llm.connect()


        connection = await self.edge.join(self, call)
        self._connection = connection

        # Attach fallback pc.on('track') handler ASAP to avoid missing early remote video tracks
        try:
            if not self._pc_track_handler_attached:
                base_pc = getattr(self._connection, "subscriber_pc", None)
                pc = None
                if base_pc is not None:
                    pc = (
                        getattr(base_pc, "pc", None)
                        or getattr(base_pc, "_pc", None)
                        or base_pc
                    )
                if pc is not None and hasattr(pc, "on"):
                    self.logger.info(
                        "🔗 Attaching pc.on('track') handler to subscriber peer connection (early)"
                    )
                    # Create or reuse a persistent MediaRelay to keep branches alive
                    try:
                        self._persistent_media_relay = (
                            getattr(self, "_persistent_media_relay", None)
                            or MediaRelay()
                        )
                    except Exception:
                        self._persistent_media_relay = None

                    @pc.on("track")
                    async def _on_pc_track_early(track):
                        try:
                            kind = getattr(track, "kind", None)
                            if kind == "video":
                                relay = self._persistent_media_relay
                                if relay is None:
                                    relay = MediaRelay()
                                    self._persistent_media_relay = relay
                                forward_branch = relay.subscribe(track)
                                print(
                                    f"🎥 Forwarding video frames to Realtime provider (pc.on early track) {forward_branch}"
                                )
                                if self.sts_mode and isinstance(self.llm, Realtime):
                                    await self.llm.start_video_sender(
                                        forward_branch, fps=30
                                    )
                                    self.logger.info(
                                        "🎥 Forwarding video frames to Realtime provider (pc.on early track)"
                                    )
                        except Exception as e:
                            self.logger.error(
                                f"Error handling pc.on('track') video (early): {e}"
                            )

                    self._pc_track_handler_attached = True
        except Exception:
            pass

        self._is_running = True

        connection._connection._coordinator_ws_client.on_wildcard(
            "*", lambda event_name, event: self.events.send(event)
        )

        self.logger.info(f"🤖 Agent joined call: {call.id}")

        # Set up audio and video tracks together to avoid SDP issues
        audio_track = self._audio_track if self.publish_audio else None
        video_track = self._video_track if self.publish_video else None

        if audio_track or video_track:
            await self.edge.publish_tracks(audio_track, video_track)

            # Set up event handlers for audio processing
            await self._listen_to_audio_and_video()

            # Video track detection is handled by event-based _process_track method
            # No need for polling since Stream Video events are reliable

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

            @self.edge.events.subscribe
            async def on_ended(event: CallEndedEvent):
                if not fut.done():
                    fut.set_result(None)

            await fut
        except Exception as e:
            logging.warning(f"⚠️ Error while waiting for call to end: {e}")
            # Don't raise the exception, just log it and continue cleanup

    def send(self, event):
        """
        Send an event through the agent's event system.

        This is a convenience method that calls agent.events.send().
        Use this for consistency with agent.events.subscribe().

        Args:
            event: The event to send
        """
        self.events.send(event)

    async def say(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Make the agent say something by emitting an event.

        Args:
            text: The text for the agent to say
            metadata: Optional metadata to include with the event
        """
        user_id = self.agent_user.id
        self.conversation.add_message(Message(content=text, user_id=user_id))

        # Emit agent say event using the send method
        self.send(
            events.AgentSayEvent(
                plugin_name="agent", text=text, user_id=user_id, metadata=metadata or {}
            )
        )

    async def send_audio(self, pcm):
        # TODO: stream & buffer
        if self._audio_track is not None:
            await self._audio_track.send_audio(pcm)

    async def create_user(self):
        response = await self.edge.create_user(self.agent_user)
        return response

    async def _handle_output_text_delta(self, event: StandardizedTextDeltaEvent):
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
                initial_content=event.delta,
            )
        else:
            self.conversation.append_to_message(
                self._agent_conversation_handle, event.delta
            )

    async def _handle_after_response(self, llm_response: LLMResponseEvent):
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

        # Trigger TTS directly instead of through event system
        if llm_response.text and llm_response.text.strip():
            await self.tts.send(llm_response.text)

    def _on_vad_audio(self, event: VADAudioEvent):
        self.logger.info(f"Vad audio event {self._truncate_for_logging(event)}")

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
                    error_message=str(e),
                    error=e,
                )
            )
            self.logger.error(f"Error in agent say: {e}")

    def _setup_mcp_servers(self):
        """Set up MCP servers if any are configured."""
        if self.mcp_servers:
            self.logger.info(f"🔌 Setting up {len(self.mcp_servers)} MCP server(s)")
            for i, server in enumerate(self.mcp_servers):
                self.logger.info(f"  {i + 1}. {server.__class__.__name__}")
        else:
            self.logger.debug("No MCP servers configured")

    def _setup_turn_detection(self):
        if self.turn_detection:
            # TODO: this subscriptions should be in plugin and just merged when
            # plugin is registered, each plugin should have access to agent
            self.logger.info("🎙️ Setting up turn detection listeners")
            self.events.subscribe(self._on_turn_started)
            self.events.subscribe(self._on_turn_ended)
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
            if self.turn_detection is not None:
                await self.turn_detection.process_audio(pcm, participant.user_id)

            await self._reply_to_audio(pcm, participant)

        # Always listen to remote video tracks so we can forward frames to Realtime providers

        @self.edge.events.subscribe
        async def on_track(event: TrackAddedEvent):
            track_id = event.track_id
            track_type = event.track_type
            user = event.user
            task = asyncio.create_task(self._process_track(track_id, track_type, user))
            task.add_done_callback(_log_task_exception)

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
                # TODO: this behaviour should be easy to change in the agent class
                task = asyncio.create_task(self.llm.simple_audio_response(pcm_data))
                #task.add_done_callback(lambda t: print(f"Task (send_audio_pcm) error: {t.exception()}"))
            else:
                # Process audio through STT
                if self.stt:
                    self.logger.debug(f"🎵 Processing audio from {participant}")
                    await self.stt.process_audio(pcm_data, participant)

    async def _process_track(self, track_id: str, track_type: str, participant):
        # Only process video tracks - track_type might be string, enum or numeric (2 for video)
        self.logger.info(
            f"🎥VDP: Checking track type: {track_type} vs {TrackType.TRACK_TYPE_VIDEO}"
        )
        if track_type not in ("video", TrackType.TRACK_TYPE_VIDEO, 2):
            self.logger.warning(
                f"🎥VDP: EARLY EXIT - Ignoring non-video track: {track_type} (expected: video, {TrackType.TRACK_TYPE_VIDEO}, or 2)"
            )
            return

        track = self.edge.add_track_subscriber(track_id)
        if not track:
            self.logger.error(
                f"🎥VDP: EARLY EXIT - Failed to subscribe to track: {track_id}"
            )
            return

        self.logger.info(
            f"🎥VDP: Track subscription successful, validating video track..."
        )

        # Determine if this is a video track using both reported kind and original type
        is_video_type = track_type in ("video", TrackType.TRACK_TYPE_VIDEO, 2)
        kind = getattr(track, "kind", None)
        is_video_kind = kind == "video"

        self.logger.info(
            f"🎥VDP: Track validation - is_video_type={is_video_type}, kind='{kind}', is_video_kind={is_video_kind}"
        )

        if not (is_video_kind or is_video_type):
            self.logger.warning(
                f"🎥VDP: EARLY EXIT - Ignoring non-video track after subscribe: kind={kind} original_type={track_type}"
            )
            return

        try:
            self.logger.info(
                f"🎥VDP: ✅ Subscribed to track: {track_id}, kind={getattr(track, 'kind', None)}, class={track.__class__.__name__}"
            )
        except Exception:
            self.logger.info(f"🎥VDP: ✅ Subscribed to track: {track_id}")

        # Give the track a moment to be ready
        self.logger.info(f"🎥VDP: Waiting for track to be ready...")
        await asyncio.sleep(0.5)

        # If Realtime provider supports video, forward frames upstream once per track
        if self.sts_mode:
            try:
                await self.llm._watch_video_track(track)
                self.logger.info("🎥 Forwarding video frames to Realtime provider")

            except Exception as e:
                self.logger.error(f"🎥VDP: ❌ Failed to start OpenAI video sender: {e}")
                self.logger.error(f"🎥VDP: Exception type: {type(e).__name__}")
                import traceback

                self.logger.error(
                    f"🎥VDP: Exception traceback: {traceback.format_exc()}"
                )
        else:
            self.logger.warning(
                f"🎥VDP: STS mode check failed - sts_mode={self.sts_mode}, llm_type={type(self.llm).__name__}"
            )
            self.logger.warning(
                f"🎥VDP: isinstance(self.llm, Realtime) = {isinstance(self.llm, Realtime)}"
            )

        hasImageProcessers = len(self.image_processors) > 0
        self.logger.info(
            f"📸 Has image processors: {hasImageProcessers}, count: {len(self.image_processors)}"
        )

        self.logger.info(
            f"📸 Starting video processing loop for track {track_id} {participant.user_id} {participant.name}"
        )

        # Enhanced video processing with timeout limits and error recovery
        timeout_errors = 0
        max_timeout_errors = 10  # Circuit breaker threshold
        base_timeout = 5.0
        consecutive_errors = 0
        max_consecutive_errors = 5

        self.logger.info(
            f"🎥VDP: Starting robust video processing loop (max_timeouts={max_timeout_errors})"
        )

        while True:
            # Periodic health check
            try:
                self._check_track_health()
            except AttributeError as e:
                # Handle case where attributes haven't been initialized yet
                if "_last_health_check" in str(
                    e
                ) or "_track_health_check_interval" in str(e):
                    self.logger.debug(
                        "🎥VDP: Health check attributes not initialized yet, skipping"
                    )
                    break
                else:
                    raise

            try:
                # Adaptive timeout based on error count
                current_timeout = base_timeout * (1.5 ** min(timeout_errors, 3))

                # Track frame processing timing
                frame_request_start = time.monotonic()

                # TODO: evaluate if this makes sense or not...
                #video_frame = await asyncio.wait_for(processing_branch.recv(), timeout=current_timeout)
                video_frame = await track.recv()
                frame_request_end = time.monotonic()
                frame_request_duration = frame_request_end - frame_request_start

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

                    # video processors
                    for processor in self.video_processors:
                        try:
                            await processor.process_video(track, participant.user_id)
                        except Exception as e:
                            self.logger.error(
                                f"Error in video processor {type(processor).__name__}: {e}"
                            )
                else:
                    self.logger.warning("🎥VDP: Received empty frame")
                    consecutive_errors += 1

            except asyncio.TimeoutError:
                timeout_errors += 1
                consecutive_errors += 1

                self.logger.warning(
                    f"🎥VDP: Timeout #{timeout_errors} (timeout={current_timeout:.1f}s, consecutive_errors={consecutive_errors})"
                )

                # Circuit breaker: stop processing if too many timeouts
                if timeout_errors >= max_timeout_errors:
                    self.logger.error(
                        f"🎥VDP: Circuit breaker triggered - too many timeouts ({timeout_errors}), stopping track processing"
                    )
                    break

                # Circuit breaker: stop processing if too many consecutive errors
                if consecutive_errors >= max_consecutive_errors:
                    self.logger.error(
                        f"🎥VDP: Circuit breaker triggered - too many consecutive errors ({consecutive_errors}), stopping track processing"
                    )
                    break

                # Exponential backoff for timeout errors
                backoff_delay = min(2.0 ** min(timeout_errors, 5), 30.0)
                self.logger.debug(
                    f"🎥VDP: Applying backoff delay: {backoff_delay:.1f}s"
                )
                await asyncio.sleep(backoff_delay)

            except Exception as e:
                consecutive_errors += 1
                self.logger.error(
                    f"🎥VDP: Error receiving track: {e} - {type(e)} (consecutive_errors={consecutive_errors})"
                )

                # Circuit breaker: stop processing on persistent errors
                if consecutive_errors >= max_consecutive_errors:
                    self.logger.error(
                        f"🎥VDP: Circuit breaker triggered - too many consecutive errors ({consecutive_errors}), stopping track processing"
                    )
                    break

                # Short delay for non-timeout errors
                await asyncio.sleep(0.5)

        # Cleanup and logging
        self.logger.info(f"🎥VDP: Video processing loop ended for track {track_id} - timeouts: {timeout_errors}, consecutive_errors: {consecutive_errors}")

        # Remove track from active tracking
        self._remove_track(track_id)

    def _on_turn_started(self, event: TurnEventData) -> None:
        """Handle when a participant starts their turn."""
        # TODO: Implement TTS pause/resume functionality
        # For now, TTS will continue playing - this should be improved
        self.logger.info(
            f"👉 Turn started - participant speaking {event.speaker_id} : {event.confidence}"
        )

    def _on_turn_ended(self, event: TurnEventData) -> None:
        """Handle when a participant ends their turn."""
        self.logger.info(
            f"👉 Turn ended - participant {event.speaker_id} finished (duration: {event.confidence})"
        )

    def _register_track(self, track_id: str, track_info: Dict) -> None:
        """Register a track for lifecycle management."""
        self._active_tracks[track_id] = {
            **track_info,
            "registered_at": time.monotonic(),
            "last_health_check": time.monotonic(),
        }
        self.logger.info(f"🎥VDP: Registered track {track_id} for lifecycle management")

    def _remove_track(self, track_id: str) -> None:
        """Remove a track from lifecycle management."""
        if track_id in self._active_tracks:
            track_info = self._active_tracks.pop(track_id)
            duration = time.monotonic() - track_info["registered_at"]
            self.logger.info(
                f"🎥VDP: Removed track {track_id} from lifecycle management (duration: {duration:.1f}s)"
            )

    def _check_track_health(self) -> None:
        """Check health of all active tracks."""
        now = time.monotonic()

        # Only check health periodically
        if now - self._last_health_check < self._track_health_check_interval:
            return

        self._last_health_check = now
        unhealthy_tracks = []

        for track_id, track_info in self._active_tracks.items():
            try:
                # Check if track has health status method
                if hasattr(track_info.get("forwarding_track"), "get_health_status"):
                    health_status = track_info["forwarding_track"].get_health_status()
                    if not health_status["is_healthy"]:
                        unhealthy_tracks.append(track_id)
                        self.logger.warning(
                            f"🎥VDP: Track {track_id} is unhealthy: {health_status}"
                        )
                else:
                    # Basic health check - if track has been active for too long without updates
                    if now - track_info["last_health_check"] > 60.0:  # 1 minute
                        unhealthy_tracks.append(track_id)
                        self.logger.warning(
                            f"🎥VDP: Track {track_id} has not been checked for 60+ seconds"
                        )

                # Update last health check time
                track_info["last_health_check"] = now

            except Exception as e:
                self.logger.error(
                    f"🎥VDP: Error checking health of track {track_id}: {e}"
                )
                unhealthy_tracks.append(track_id)

        # Log summary
        if unhealthy_tracks:
            self.logger.warning(
                f"🎥VDP: Found {len(unhealthy_tracks)} unhealthy tracks: {unhealthy_tracks}"
            )
        else:
            self.logger.debug(
                f"🎥VDP: All {len(self._active_tracks)} tracks are healthy"
            )

    def get_track_health_summary(self) -> Dict[str, Any]:
        """Get a summary of all track health status."""
        summary = {
            "total_tracks": len(self._active_tracks),
            "healthy_tracks": 0,
            "unhealthy_tracks": 0,
            "track_details": {},
        }

        for track_id, track_info in self._active_tracks.items():
            try:
                if hasattr(track_info.get("forwarding_track"), "get_health_status"):
                    health_status = track_info["forwarding_track"].get_health_status()
                    summary["track_details"][track_id] = health_status
                    if health_status["is_healthy"]:
                        summary["healthy_tracks"] += 1
                    else:
                        summary["unhealthy_tracks"] += 1
                else:
                    summary["track_details"][track_id] = {"status": "unknown"}
            except Exception as e:
                summary["track_details"][track_id] = {"error": str(e)}
                summary["unhealthy_tracks"] += 1

        return summary

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
            self.conversation.replace_message(
                self._user_conversation_handle, event.text
            )
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
        """Connect to all configured MCP servers and register their tools."""
        if not self.mcp_servers:
            return

        self.logger.info(f"🔌 Connecting to {len(self.mcp_servers)} MCP server(s)")

        for i, server in enumerate(self.mcp_servers):
            try:
                self.logger.info(
                    f"  Connecting to MCP server {i + 1}/{len(self.mcp_servers)}: {server.__class__.__name__}"
                )
                await server.connect()
                self.logger.info(
                    f"  ✅ Connected to MCP server {i + 1}/{len(self.mcp_servers)}"
                )

                # Register MCP tools with the LLM's function registry
                await self._register_mcp_tools(i, server)

            except Exception as e:
                self.logger.error(
                    f"  ❌ Failed to connect to MCP server {i + 1}/{len(self.mcp_servers)}: {e}"
                )
                # Continue with other servers even if one fails

    async def _disconnect_mcp_servers(self):
        """Disconnect from all configured MCP servers."""
        if not self.mcp_servers:
            return

        self.logger.info(f"🔌 Disconnecting from {len(self.mcp_servers)} MCP server(s)")

        for i, server in enumerate(self.mcp_servers):
            try:
                self.logger.info(
                    f"  Disconnecting from MCP server {i + 1}/{len(self.mcp_servers)}: {server.__class__.__name__}"
                )
                await server.disconnect()
                self.logger.info(
                    f"  ✅ Disconnected from MCP server {i + 1}/{len(self.mcp_servers)}"
                )
            except Exception as e:
                self.logger.error(
                    f"  ❌ Error disconnecting from MCP server {i + 1}/{len(self.mcp_servers)}: {e}"
                )
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
                    self.logger.error(
                        f"Error getting tools from MCP server {server.__class__.__name__}: {e}"
                    )

        return tools

    async def call_mcp_tool(
        self, server_index: int, tool_name: str, arguments: Dict[str, Any]
    ) -> Any:
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

    async def _register_mcp_tools(self, server_index: int, server: MCPBaseServer):
        """Register tools from an MCP server with the LLM's function registry.

        Args:
            server_index: Index of the MCP server in the mcp_servers list
            server: The connected MCP server
        """
        try:
            # Get tools from the MCP server
            mcp_tools = await server.list_tools()
            self.logger.info(
                f"  📋 Found {len(mcp_tools)} tools from MCP server {server_index + 1}"
            )

            # Register each tool with the function registry
            for tool in mcp_tools:
                try:
                    # Create a wrapper function for the MCP tool
                    tool_wrapper = MCPToolConverter.create_mcp_tool_wrapper(
                        server_index, tool.name, self
                    )

                    # Register the tool with the LLM's function registry
                    self.llm.function_registry.register(
                        name=f"mcp_{server_index}_{tool.name}",
                        description=tool.description or f"MCP tool: {tool.name}",
                    )(tool_wrapper)

                    self.logger.info(f"    ✅ Registered tool: {tool.name}")

                except Exception as e:
                    self.logger.error(
                        f"    ❌ Failed to register tool {tool.name}: {e}"
                    )
                    # Continue with other tools even if one fails

        except Exception as e:
            self.logger.error(
                f"  ❌ Failed to get tools from MCP server {server_index + 1}: {e}"
            )
