import json
from typing import Any, Optional, List, Dict

from getstream.video.rtc.audio_track import AudioStreamTrack
from openai.types.realtime import RealtimeSessionCreateRequestParam, ResponseAudioTranscriptDoneEvent, \
    InputAudioBufferSpeechStartedEvent

from vision_agents.core.llm import realtime
from vision_agents.core.llm.llm_types import ToolSchema
import logging
from dotenv import load_dotenv
from getstream.video.rtc.track_util import PcmData
from .rtc_manager import RTCManager

from vision_agents.core.edge.types import Participant
from vision_agents.core.processors import Processor

load_dotenv()

logger = logging.getLogger(__name__)


"""
TODO
- support passing full client options in __init__. at that's not possible. 
- review either SessionCreateParams or RealtimeSessionCreateRequestParam 
- send video should depend on if the RTC connection with stream is sending video.
"""

client = RealtimeSessionCreateRequestParam


class Realtime(realtime.Realtime):
    """
    OpenAI Realtime API implementation for real-time AI audio and video communication over WebRTC.

    Extends the base Realtime class with WebRTC-based audio and optional video
    streaming to OpenAI's servers. Supports speech-to-speech conversation, text
    messaging, multimodal interactions, and function calling with MCP support.

    Args:
        model: OpenAI model to use (e.g., "gpt-realtime").
        voice: Voice for audio responses (e.g., "marin", "alloy").
        send_video: Enable video streaming capabilities. Defaults to False.

        This class uses:
        - RTCManager to handle WebRTC connection and media streaming.
        - Output track to forward audio and video to the remote participant.
        - Function calling support for real-time tool execution.
        - MCP integration for external service access.

    """
    def __init__(self, model: str = "gpt-realtime", voice: str = "marin", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.voice = voice
        # TODO: send video should depend on if the RTC connection with stream is sending video.
        self.rtc = RTCManager(self.model, self.voice, True)
        # audio output track?
        self.output_track = AudioStreamTrack(
            framerate=48000, stereo=True, format="s16"
        )

    async def connect(self):
        """Establish the WebRTC connection to OpenAI's Realtime API.

        Sets up callbacks and connects to OpenAI's servers. Emits connected event
        with session configuration when ready.
        """
        instructions: Optional[str] = None
        if hasattr(self, "parsed_instructions") and self.parsed_instructions:
            instructions = self._build_enhanced_instructions()
        elif getattr(self, "instructions", None):
            instructions = self.instructions

        self.rtc.instructions = instructions
        
        # Wire callbacks so we can emit audio/events upstream
        self.rtc.set_event_callback(self._handle_openai_event)
        self.rtc.set_audio_callback(self._handle_audio_output)
        await self.rtc.connect()
        
        # Register tools with OpenAI realtime if available
        await self._register_tools_with_openai_realtime()
        
        # Emit connected/ready
        self._emit_connected_event(
            session_config={"model": self.model, "voice": self.voice},
            capabilities=["text", "audio", "function_calling"],
        )

    async def simple_response(self, text: str, processors: Optional[List[Processor]] = None,
      participant: Optional[Participant] = None):
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

    async def simple_audio_response(self, audio: PcmData):
        """Send a single PCM audio frame to the OpenAI Realtime session.

        The audio should be raw PCM matching the realtime session's expected
        format (typically 48 kHz mono, 16-bit). For continuous audio capture,
        call this repeatedly with consecutive frames.

        Args:
            audio: PCM audio frame to forward upstream.
        """
        await self.rtc.send_audio_pcm(audio)

    async def request_session_info(self) -> None:
        """Request session information from the OpenAI API.

        Delegates to the RTC manager to query session metadata.
        """
        await self.rtc.request_session_info()

    async def _close_impl(self):
        """Close the OpenAI Realtime session.

        Delegates cleanup to the RTC manager. Called by the base class close() method.
        """
        await self.rtc.close()

    async def _handle_openai_event(self, event: dict) -> None:
        """Process events received from the OpenAI Realtime API.

        Handles OpenAI event types and emits standardized events.

        Args:
            event: Raw event dictionary from OpenAI API.

        Event Handling:
            - response.audio_transcript.done: Emits transcript/response events
            - input_audio_buffer.speech_started: Flushes output audio track
            - response.tool_call: Handles tool calls from OpenAI realtime

        Note:
            Registered as callback with RTC manager.
        """
        et = event.get("type")
        if et == "response.audio_transcript.done":
            transcript_event: ResponseAudioTranscriptDoneEvent = ResponseAudioTranscriptDoneEvent.model_validate(event)
            self._emit_transcript_event(text=transcript_event.transcript, user_metadata={"role": "assistant", "source": "openai"})
            self._emit_response_event(text=transcript_event.transcript, response_id=transcript_event.response_id, is_complete=True, conversation_item_id=transcript_event.item_id)
        elif et == "input_audio_buffer.speech_started":
            # Validate event but don't need to store it
            InputAudioBufferSpeechStartedEvent.model_validate(event)
            await self.output_track.flush()
        elif et == "response.output_item.added":
            # Check if this is a function call
            item = event.get("item", {})
            if item.get("type") == "function_call":
                await self._handle_tool_call_event(event)
        elif et == "response.tool_call":
            # Handle tool calls from OpenAI realtime
            await self._handle_tool_call_event(event)

    async def _handle_audio_output(self, audio_bytes: bytes) -> None:
        """Process audio output received from the OpenAI API.

        Forwards audio data to the output track for playback and emits audio output event.

        Args:
            audio_bytes: Raw audio data bytes from OpenAI session.

        Note:
            Registered as callback with RTC manager.
        """
        # Emit audio output event
        self._emit_audio_output_event(
            audio_data=audio_bytes,
            sample_rate=48000,  # OpenAI Realtime uses 48kHz
        )
        
        # Forward audio to output track for playback
        await self.output_track.write(audio_bytes)

    async def _watch_video_track(self, track, **kwargs) -> None:
        shared_forwarder = kwargs.get('shared_forwarder')
        await self.rtc.start_video_sender(track, self.fps, shared_forwarder=shared_forwarder)

    async def _stop_watching_video_track(self) -> None:
        await self.rtc.stop_video_sender()

    async def _handle_tool_call_event(self, event: dict) -> None:
        """Handle tool call events from OpenAI realtime.
        
        Args:
            event: Tool call event from OpenAI realtime API
        """
        try:
            # Handle both event structures
            if event.get("type") == "response.output_item.added":
                item = event.get("item", {})
                tool_call_data = item
            else:
                tool_call_data = event.get("tool_call", {})
                
            if not tool_call_data:
                logger.warning("Received tool call event without tool_call data")
                return
                
            # Extract tool call details
            tool_call = {
                "type": "tool_call",
                "id": tool_call_data.get("call_id"),
                "name": tool_call_data.get("name", "unknown"),
                "arguments_json": tool_call_data.get("arguments", {})
            }
            
            logger.info(f"Executing tool call: {tool_call['name']} with args: {tool_call['arguments_json']}")
            
            # Execute using existing tool execution infrastructure
            tc, result, error = await self._run_one_tool(tool_call, timeout_s=30)
            
            # Prepare response data
            if error:
                response_data = {"error": str(error)}
                logger.error(f"Tool call {tool_call['name']} failed: {error}")
            else:
                # Ensure response is a dictionary for OpenAI realtime
                if not isinstance(result, dict):
                    response_data = {"result": result}
                else:
                    response_data = result
                logger.info(f"Tool call {tool_call['name']} succeeded: {response_data}")
            
            # Send tool response back to OpenAI realtime session
            await self._send_tool_response(tool_call["id"], response_data)
            
        except Exception as e:
            logger.error(f"Error handling tool call event: {e}")
            # Send error response back
            call_id = None
            if event.get("type") == "response.output_item.added":
                call_id = event.get("item", {}).get("call_id")
            else:
                call_id = event.get("tool_call", {}).get("call_id")
            await self._send_tool_response(call_id, {"error": str(e)})

    async def _send_tool_response(self, call_id: Optional[str], response_data: Dict[str, Any]) -> None:
        """Send tool response back to OpenAI realtime session.
        
        Args:
            call_id: The call ID from the original tool call
            response_data: The response data to send back
        """
        if not call_id:
            logger.warning("Cannot send tool response without call_id")
            return
            
        try:
            # Convert response to string for OpenAI realtime
            response_str = self._sanitize_tool_output(response_data)
            
            # Send tool response event
            event = {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": response_str
                }
            }
            
            await self.rtc._send_event(event)
            logger.info(f"Sent tool response for call_id {call_id}")
            
            # Trigger a new response to continue the conversation with audio
            # This ensures the AI responds with audio after receiving the tool result
            await self.rtc._send_event({
                "type": "response.create",
                "response": {
                    "modalities": ["text", "audio"],
                    "instructions": "Please respond to the user with the tool results in a conversational way."
                }
            })
            
        except Exception as e:
            logger.error(f"Failed to send tool response: {e}")

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

    def _convert_tools_to_openai_realtime_format(self, tools: List[ToolSchema]) -> List[Dict[str, Any]]:
        """Convert ToolSchema objects to OpenAI realtime format.
        
        Args:
            tools: List of ToolSchema objects from the function registry
            
        Returns:
            List of tools in OpenAI realtime format
        """
        out = []
        for t in tools or []:
            name = t.get("name", "unnamed_tool")
            description = t.get("description", "") or ""
            params = t.get("parameters_schema") or t.get("parameters") or {}
            if not isinstance(params, dict):
                params = {}
            params.setdefault("type", "object")
            params.setdefault("properties", {})
            params.setdefault("additionalProperties", False)

            out.append({
                "type": "function",
                "name": name,
                "description": description,
                "parameters": params,
            })
        return out

    async def _register_tools_with_openai_realtime(self) -> None:
        """Register available tools with OpenAI realtime session.
        
        This method registers all available functions and MCP tools with the
        OpenAI realtime session so they can be called during conversations.
        """
        try:
            # Get available tools from function registry
            available_tools = self.get_available_functions()
            
            if not available_tools:
                logger.info("No tools available to register with OpenAI realtime")
                return
                
            # Convert tools to OpenAI realtime format
            tools_for_openai = self._convert_tools_to_openai_realtime_format(available_tools)
            
            if not tools_for_openai:
                logger.info("No tools converted for OpenAI realtime")
                return
                
            # Send tools configuration to OpenAI realtime
            tools_event = {
                "type": "session.update",
                "session": {
                    "tools": tools_for_openai
                }
            }
            
            await self.rtc._send_event(tools_event)
            logger.info(f"Registered {len(tools_for_openai)} tools with OpenAI realtime")
            
        except Exception as e:
            logger.error(f"Failed to register tools with OpenAI realtime: {e}")
            # Don't raise the exception - tool registration failure shouldn't break the connection
