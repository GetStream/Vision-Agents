import json
import logging
from typing import Any, Optional, List, Dict, Union

import aiortc
from openai import AsyncOpenAI
from openai.types.realtime import (
    RateLimitsUpdatedEvent,
    SessionUpdatedEvent,
    RealtimeSessionCreateRequestParam,
    RealtimeAudioConfigParam,
    RealtimeAudioConfigOutputParam,
    RealtimeAudioConfigInputParam,
    AudioTranscriptionParam,
    ResponseAudioTranscriptDoneEvent,
    InputAudioBufferSpeechStartedEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ResponseDoneEvent,
    SessionCreatedEvent,
    RealtimeSessionCreateRequest,
    RealtimeTranscriptionSessionCreateRequest,
)
from openai.types.realtime.realtime_transcription_session_audio_input_turn_detection_param import (
    SemanticVad,
)
from dotenv import load_dotenv
from getstream.video.rtc.track_util import PcmData

from vision_agents.core.llm import realtime
from vision_agents.core.edge.types import Participant
from vision_agents.core.processors import Processor
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.instructions import Instructions

from .rtc_manager import RTCManager
from .tool_utils import convert_tools_to_openai_format, parse_tool_arguments

load_dotenv()

logger = logging.getLogger(__name__)

"""
TODO: Future improvements
- Reconnection is currently not easy to do with OpenAI realtime.
"""


class Realtime(realtime.Realtime):
    """
    OpenAI Realtime API implementation for real-time AI audio and video communication over WebRTC.

    Extends the base Realtime class with WebRTC-based audio and optional video
    streaming to OpenAI's servers. Supports speech-to-speech conversation, text
    messaging, multimodal interactions, and function calling with MCP support.

    Args:
        model: OpenAI model to use (e.g., "gpt-realtime").
        voice: Voice for audio responses (e.g., "marin", "alloy").
        realtime_session: Configure RealtimeSessionCreateRequestParam

        api_key: Optionally specify an API key
        client: pass your own AsyncOpenAI client

        This class uses:
        - RTCManager to handle WebRTC connection and media streaming.
        - Output track to forward audio and video to the remote participant.
        - Function calling support for real-time tool execution.
        - MCP integration for external service access.

    """

    def __init__(
        self,
        model: str = "gpt-realtime",
        api_key: Optional[str] = None,
        voice: str = "marin",
        client: Optional[AsyncOpenAI] = None,
        fps: int = 1,
        realtime_session: Optional[RealtimeSessionCreateRequestParam] = None,
        send_video: bool = True,
    ):
        super().__init__(fps)
        self.model = model
        self.voice = voice

        self.realtime_session: RealtimeSessionCreateRequestParam = (
            realtime_session or RealtimeSessionCreateRequestParam(type="realtime")
        )
        self.realtime_session["model"] = self.model

        # Set audio and output if they are None
        if self.realtime_session.get("audio") is None:
            self.realtime_session["audio"] = RealtimeAudioConfigParam(
                input=RealtimeAudioConfigInputParam(
                    transcription=AudioTranscriptionParam(
                        model="gpt-4o-mini-transcribe"
                    ),
                    turn_detection=SemanticVad(type="semantic_vad"),
                )
            )
        if self.realtime_session["audio"].get("output") is None:
            self.realtime_session["audio"]["output"] = RealtimeAudioConfigOutputParam()
        self.realtime_session["audio"]["output"]["voice"] = self.voice

        # Track pending tool calls: item_id -> {call_id, name, argument_parts: []}
        # We accumulate argument deltas until response.output_item.done
        self._pending_tool_calls: Dict[str, Dict[str, Any]] = {}

        # Store current session and rate limits
        self.current_session: Optional[
            Union[
                RealtimeSessionCreateRequest, RealtimeTranscriptionSessionCreateRequest
            ]
        ] = None
        self.current_rate_limits: Optional[RateLimitsUpdatedEvent] = None

        # create the client
        if client is not None:
            self.client = client
        elif api_key is not None and api_key != "":
            self.client = AsyncOpenAI(api_key=api_key)
        else:
            self.client = AsyncOpenAI()  # will get it from the env vars

        # Start the realtime connection manager
        self.rtc = RTCManager(
            realtime_session=self.realtime_session,
            client=self.client,
            send_video=send_video,
        )

    async def connect(self):
        """Establish the WebRTC connection to OpenAI's Realtime API.

        Sets up callbacks and connects to OpenAI's servers. Emits connected event
        with session configuration when ready.
        """
        # Add tools to the session config BEFORE connecting
        # Tools are registered via @llm.register_function() before connect() is called
        available_tools = self.get_available_functions()
        if available_tools:
            tools_for_openai = convert_tools_to_openai_format(
                available_tools, for_realtime=True
            )
            self.realtime_session["tools"] = tools_for_openai  # type: ignore[typeddict-item]
            logger.info(
                f"Added {len(tools_for_openai)} tools to session config: {[t['name'] for t in tools_for_openai]}"
            )

        # Wire callbacks so we can emit audio/events upstream
        self.rtc.set_event_callback(self._handle_openai_event)
        self.rtc.set_audio_callback(self._handle_audio_output)
        await self.rtc.connect()

        # Emit connected/ready
        self._emit_connected_event(
            session_config={"model": self.model, "voice": self.voice},
            capabilities=["text", "audio", "function_calling"],
        )

    async def simple_response(
        self,
        text: str,
        processors: Optional[List[Processor]] = None,
        participant: Optional[Participant] = None,
    ):
        """Send a simple text input to the OpenAI Realtime session.

        This is a convenience wrapper that forwards a text prompt upstream via
        the underlying realtime connection. It does not stream partial deltas
        back; callers should subscribe to the provider's events to receive
        responses.

        Args:
            text: Text prompt to send.
            processors: Optional processors list (not used here; included for
                interface parity with the core `LLM` API).
            participant: Optional participant metadata (ignored here).
        """
        await self.rtc.send_text(text)

    async def simple_audio_response(
        self, audio: PcmData, participant: Optional[Participant] = None
    ):
        """Send a single PCM audio frame to the OpenAI Realtime session.

        The audio should be raw PCM matching the realtime session's expected
        format (typically 48 kHz mono, 16-bit). For continuous audio capture,
        call this repeatedly with consecutive frames.

        Args:
            audio: PCM audio frame to forward upstream.
            participant: Optional participant information for the audio source.
        """
        # Track current participant for user speech transcription events
        self._current_participant = participant
        await self.rtc.send_audio_pcm(audio)

    async def close(self):
        await self.rtc.close()

    async def _handle_openai_event(self, event: dict) -> None:
        """Process events received from the OpenAI Realtime API.

        Handles OpenAI event types and emits standardized events.

        Args:
            event: Raw event dictionary from OpenAI API.

        Event Handling:
            - response.audio_transcript.done: Emits agent speech transcription
            - conversation.item.input_audio_transcription.completed: Emits user speech transcription
            - input_audio_buffer.speech_started: Flushes output audio track
            - response.tool_call: Handles tool calls from OpenAI realtime

        Note:
            Registered as callback with RTC manager.
        """
        et = event.get("type")

        # code here is weird because OpenAI does something strange
        # see issue: https://github.com/openai/openai-python/issues/2698
        # as a workaround we copy the event and normalize the type to response.output_audio_transcript.done so that
        # ResponseAudioTranscriptDoneEvent.model_validate is happy
        if et in [
            "response.audio_transcript.done",
            "response.output_audio_transcript.done",
        ]:
            # Create a copy and normalize the type field
            event_copy = event.copy()
            event_copy["type"] = "response.output_audio_transcript.done"
            transcript_event: ResponseAudioTranscriptDoneEvent = (
                ResponseAudioTranscriptDoneEvent.model_validate(event_copy)
            )
            self._emit_agent_speech_transcription(
                text=transcript_event.transcript, original=event
            )
            self._emit_response_event(
                text=transcript_event.transcript,
                response_id=transcript_event.response_id,
                is_complete=True,
                conversation_item_id=transcript_event.item_id,
            )
        elif et == "conversation.item.created":
            # Conversation item created - no action needed
            # Participant tracking is handled via _current_participant in simple_audio_response
            pass
        elif et == "conversation.item.added":
            # Conversation item was added to the conversation
            pass
        elif et == "conversation.item.done":
            # Conversation item is complete
            pass
        elif et == "conversation.item.input_audio_transcription.completed":
            # User input audio transcription completed
            user_transcript_event: ConversationItemInputAudioTranscriptionCompletedEvent = ConversationItemInputAudioTranscriptionCompletedEvent.model_validate(
                event
            )
            # _current_participant is kept up-to-date in simple_audio_response
            # so it will be used by _emit_user_speech_transcription
            self._emit_user_speech_transcription(
                text=user_transcript_event.transcript, original=event
            )
        elif et == "input_audio_buffer.speech_started":
            # Validate event but don't need to store it
            InputAudioBufferSpeechStartedEvent.model_validate(event)
            # await self.output_track.flush()
        elif et == "response.output_item.added":
            # Start tracking a function call (arguments will stream in)
            item = event.get("item", {})
            if item.get("type") == "function_call":
                item_id = item.get("id")
                if item_id:
                    self._pending_tool_calls[item_id] = {
                        "call_id": item.get("call_id"),
                        "name": item.get("name", "unknown"),
                        "argument_parts": [],
                    }
                    logger.debug(
                        f"Started tracking tool call: {item.get('name')} (item_id={item_id})"
                    )
        elif et == "response.function_call_arguments.delta":
            # Accumulate argument deltas
            item_id = event.get("item_id")
            delta = event.get("delta", "")
            if item_id and item_id in self._pending_tool_calls:
                self._pending_tool_calls[item_id]["argument_parts"].append(delta)
        elif et == "response.function_call_arguments.done":
            # Arguments complete - execute the tool call
            item_id = event.get("item_id")
            if item_id and item_id in self._pending_tool_calls:
                await self._execute_pending_tool_call(item_id)
        elif et == "response.output_item.done":
            # Fallback: if we have a pending tool call for this item, execute it
            item = event.get("item", {})
            item_id = item.get("id")
            if item_id and item_id in self._pending_tool_calls:
                await self._execute_pending_tool_call(item_id)
        elif et == "response.created":
            pass
        elif et == "session.created":
            session_event = SessionCreatedEvent(**event)
            self.current_session = session_event.session
            logger.info("Session created %s", event)
        elif et == "rate_limits.updated":
            self.current_rate_limits = RateLimitsUpdatedEvent(**event)
        elif et == "response.done":
            response_done_event = ResponseDoneEvent.model_validate(event)

            if response_done_event.response.status == "failed":
                raise Exception(
                    "OpenAI realtime failure %s", response_done_event.response
                )
        elif et == "session.updated":
            # Update session with new data
            session_updated_event = SessionUpdatedEvent(**event)
            self.current_session = session_updated_event.session
            logger.info("Session updated %s", event)
        elif et == "response.content_part.added":
            # Content part added to response - logged for debugging
            pass
        elif et == "response.audio_transcript.delta":
            # Streaming transcript delta - logged at debug level to avoid clutter
            pass
        elif et == "response.output_audio_transcript.delta":
            # Streaming output audio transcript delta - logged at debug level to avoid clutter
            pass
        elif et == "output_audio_buffer.started":
            # Output audio buffer started - acknowledgment of audio playback start
            pass
        elif et == "output_audio_buffer.stopped":
            # Output audio buffer stopped - acknowledgment of audio playback end
            pass
        elif et == "response.audio.done":
            # Audio generation complete for this response item
            pass
        elif et == "response.output_audio.done":
            # Output audio generation complete for this response item
            pass
        elif et == "response.content_part.done":
            # Content part complete - contains full transcript
            pass
        else:
            logger.debug(f"Unrecognized OpenAI Realtime event: {et} {event}")

    async def _handle_audio_output(self, pcm: PcmData) -> None:
        """Process audio output received from the OpenAI API.

        Forwards audio data to the output track for playback and emits audio output event.
        """

        # Emit audio output event
        self._emit_audio_output_event(
            audio_data=pcm,
        )

    async def watch_video_track(
        self,
        track: aiortc.mediastreams.MediaStreamTrack,
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        """
        Watch the video track and forward data to OpenAI Realtime API.

        Args:
            track: Video track to watch and forward.
            shared_forwarder: Optional shared VideoForwarder instance to use instead
                of creating a new one. Allows multiple consumers to share the same
                video stream.
        """
        await self.rtc.start_video_sender(
            track, self.fps, shared_forwarder=shared_forwarder
        )

    async def _stop_watching_video_track(self) -> None:
        # Video sender will be stopped when connection closes
        pass

    async def _execute_pending_tool_call(self, item_id: str) -> None:
        """Execute a pending tool call after arguments are complete.

        Args:
            item_id: The item ID of the tool call to execute
        """
        pending = self._pending_tool_calls.pop(item_id, None)
        if not pending:
            return

        call_id = pending["call_id"]
        name = pending["name"]
        arguments_str = "".join(pending["argument_parts"])
        arguments = parse_tool_arguments(arguments_str)

        tool_call = {
            "type": "tool_call",
            "id": call_id,
            "name": name,
            "arguments_json": arguments,
        }

        logger.info(f"Executing tool call: {name} with args: {arguments}")

        tc, result, error = await self._run_one_tool(tool_call, timeout_s=30)

        if error:
            response_data = {"error": str(error)}
            logger.error(f"Tool call {name} failed: {error}")
        else:
            response_data = (
                {"result": result} if not isinstance(result, dict) else result
            )
            logger.info(f"Tool call {name} succeeded: {response_data}")

        await self._send_tool_response(call_id, response_data)

    async def _send_tool_response(
        self, call_id: Optional[str], response_data: Dict[str, Any]
    ) -> None:
        """Send tool response back to OpenAI realtime session.

        Args:
            call_id: The call ID from the original tool call
            response_data: The response data to send back
        """
        if not call_id:
            logger.warning("Cannot send tool response without call_id")
            return

        response_str = self._sanitize_tool_output(response_data)

        # Send tool response event
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": response_str,
            },
        }

        await self.rtc._send_event(event)
        logger.info(f"Sent tool response for call_id {call_id}")

        # Trigger a response to continue the conversation
        # Note: We don't override modalities/instructions - let the session config apply
        await self.rtc._send_event({"type": "response.create"})

    def set_instructions(self, instructions: Instructions | str) -> None:
        super().set_instructions(instructions)
        self.realtime_session["instructions"] = self._instructions

    def _sanitize_tool_output(self, value: Any, max_chars: int = 60_000) -> str:
        """Sanitize tool output for OpenAI realtime.

        Args:
            value: The tool output to sanitize
            max_chars: Maximum characters allowed (not used in realtime mode)

        Returns:
            Sanitized string output
        """
        if isinstance(value, str):
            return value
        elif isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        else:
            return str(value)

    async def update_tools(self) -> None:
        """Update tools in the OpenAI realtime session.

        Call this method if tools are registered after connect() is called.
        Normally tools are included in the initial session config.
        """
        available_tools = self.get_available_functions()

        if not available_tools:
            logger.debug("No tools available to register with OpenAI realtime")
            return

        tools_for_openai = convert_tools_to_openai_format(
            available_tools, for_realtime=True
        )

        tools_event = {
            "type": "session.update",
            "session": {"tools": tools_for_openai},
        }

        await self.rtc._send_event(tools_event)
        logger.info(f"Updated {len(tools_for_openai)} tools in OpenAI realtime session")
