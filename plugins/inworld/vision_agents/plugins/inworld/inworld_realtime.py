"""Inworld.ai Realtime API implementation over WebRTC.

Inworld's Realtime API is OpenAI-wire-compatible — same event schema, same
session-config shape, data channel named `oai-events`. This plugin reuses
OpenAI's `openai.types.realtime` TypedDicts as the wire-format type hints.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

import aiortc.mediastreams
import httpx
from dotenv import load_dotenv
from getstream.video.rtc.track_util import PcmData
from openai.types.realtime import (
    RealtimeAudioConfigInputParam,
    RealtimeAudioConfigOutputParam,
    RealtimeAudioConfigParam,
    RealtimeSessionCreateRequestParam,
)
from openai.types.realtime.realtime_transcription_session_audio_input_turn_detection_param import (
    SemanticVad,
)
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.edge.types import Participant
from vision_agents.core.instructions import Instructions
from vision_agents.core.llm import realtime

from .rtc_manager import RTCManager
from .tool_utils import convert_tools_to_openai_format, parse_tool_arguments

load_dotenv()

logger = logging.getLogger(__name__)


class InworldRealtimeError(Exception):
    """Raised when the Inworld Realtime session fails."""


class Realtime(realtime.Realtime):
    """Inworld.ai Realtime API over WebRTC.

    Speech-to-speech conversational agent with function calling. Inworld
    proxies multiple upstream model providers; the `model` argument takes
    a provider-prefixed ID like `"openai/gpt-4o-mini"` or
    `"google-ai-studio/gemini-2.5-flash"`.

    Args:
        model: Model ID (e.g. ``"openai/gpt-4o-mini"``).
        voice: Voice for audio responses (e.g. ``"Dennis"``, ``"Clive"``).
        api_key: Inworld API key. Falls back to ``INWORLD_API_KEY`` env var.
        instructions: System prompt.
        realtime_session: Advanced escape hatch — pass a full
            ``RealtimeSessionCreateRequestParam`` to set Inworld-specific
            fields (routers, custom turn-detection, tool_choice, etc.).
        fps: Video frames per second (reserved — Inworld does not currently
            accept video input; this is kept for base-class compatibility).
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        voice: str = "Dennis",
        api_key: Optional[str] = None,
        instructions: Optional[str] = None,
        realtime_session: Optional[RealtimeSessionCreateRequestParam] = None,
        fps: int = 1,
    ):
        resolved_key = api_key or os.getenv("INWORLD_API_KEY")
        if not resolved_key:
            raise ValueError(
                "INWORLD_API_KEY environment variable must be set or api_key must be provided"
            )

        super().__init__(fps)
        self.provider_name = "inworld_realtime"
        self._api_key = resolved_key
        self.model = model
        self.voice = voice

        self.realtime_session: RealtimeSessionCreateRequestParam = (
            realtime_session or RealtimeSessionCreateRequestParam(type="realtime")
        )
        self.realtime_session["model"] = model

        if self.realtime_session.get("audio") is None:
            self.realtime_session["audio"] = RealtimeAudioConfigParam(
                input=RealtimeAudioConfigInputParam(
                    turn_detection=SemanticVad(type="semantic_vad"),
                ),
            )
        if self.realtime_session["audio"].get("output") is None:
            self.realtime_session["audio"]["output"] = RealtimeAudioConfigOutputParam()
        self.realtime_session["audio"]["output"]["voice"] = voice

        if instructions is not None:
            self.realtime_session["instructions"] = instructions

        self._pending_tool_calls: Dict[str, Dict[str, Any]] = {}

        self.current_session: Optional[Dict[str, Any]] = None
        self.current_rate_limits: Optional[Dict[str, Any]] = None

        self.rtc = RTCManager(
            api_key=self._api_key,
            realtime_session=self.realtime_session,
        )

    async def connect(self) -> None:
        """Establish the WebRTC connection to Inworld's Realtime API.

        Emits ``RealtimeErrorEvent`` on failure before re-raising so subscribers
        are notified even if the caller does not catch.
        """
        available_tools = self.get_available_functions()
        if available_tools:
            tools_for_inworld = convert_tools_to_openai_format(
                available_tools, for_realtime=True
            )
            self.realtime_session["tools"] = tools_for_inworld  # type: ignore[typeddict-item]
            logger.info(
                "Added %d tools to Inworld session config: %s",
                len(tools_for_inworld),
                [t["name"] for t in tools_for_inworld],
            )

        self.rtc.set_event_callback(self._handle_inworld_event)
        self.rtc.set_audio_callback(self._handle_audio_output)

        try:
            await self.rtc.connect()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in (401, 403):
                context = "auth"
            elif status == 429:
                context = "rate_limit"
            else:
                context = f"http_{status}"
            self._emit_error_event(exc, context=context)
            raise
        except httpx.RequestError as exc:
            self._emit_error_event(exc, context="network")
            raise
        except ConnectionError as exc:
            self._emit_error_event(exc, context="webrtc")
            raise

        self._emit_connected_event(
            session_config={"model": self.model, "voice": self.voice},
            capabilities=["text", "audio", "function_calling"],
        )

    async def simple_response(
        self,
        text: str,
        participant: Optional[Participant] = None,
    ):
        """Send a text prompt to the Inworld Realtime session."""
        await self.rtc.send_text(text)

    async def simple_audio_response(
        self, audio: PcmData, participant: Optional[Participant] = None
    ):
        """Send a single PCM audio frame to the Inworld Realtime session.

        Args:
            audio: PCM audio frame at 48 kHz mono (aiortc handles the
                Opus negotiation with Inworld's 24 kHz transcoder).
            participant: Optional participant metadata for transcription events.
        """
        self._current_participant = participant
        await self.rtc.send_audio_pcm(audio)

    async def close(self) -> None:
        await self._await_pending_tools()
        await self.rtc.close()
        self._emit_disconnected_event(reason="client_close", was_clean=True)

    async def watch_video_track(
        self,
        track: aiortc.mediastreams.MediaStreamTrack,
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        """No-op — Inworld Realtime does not accept video input."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "watch_video_track called; Inworld has no video input — ignoring"
            )

    async def stop_watching_video_track(self) -> None:
        """No-op — Inworld Realtime does not accept video input."""
        return None

    async def _handle_inworld_event(self, event: dict) -> None:
        """Dispatch events received from the Inworld Realtime API.

        Uses defensive ``.get()`` reads rather than strict pydantic
        validation against OpenAI's TypedDicts — Inworld's event schema
        drifts (e.g. ``response.done`` uses ``"text"``/``"audio"`` where
        OpenAI expects ``"input_text"``/``"output_text"``; user
        transcription events omit the ``usage`` field).
        """
        et = event.get("type")

        if et in (
            "response.audio_transcript.done",
            "response.output_audio_transcript.done",
        ):
            transcript = event.get("transcript", "")
            self._emit_agent_speech_transcription(
                text=transcript, mode="final", original=event
            )
            self._emit_response_event(
                text=transcript,
                response_id=event.get("response_id"),
                is_complete=True,
                conversation_item_id=event.get("item_id"),
            )
        elif et == "conversation.item.input_audio_transcription.completed":
            self._emit_user_speech_transcription(
                text=event.get("transcript", ""), mode="final", original=event
            )
        elif et == "input_audio_buffer.speech_started":
            await self.interrupt()
            self._emit_audio_output_done_event(interrupted=True)
        elif et == "response.output_item.added":
            item = event.get("item", {})
            if item.get("type") == "function_call":
                item_id = item.get("id")
                if item_id:
                    self._pending_tool_calls[item_id] = {
                        "call_id": item.get("call_id"),
                        "name": item.get("name", "unknown"),
                        "argument_parts": [],
                    }
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "Started tracking tool call: %s (item_id=%s)",
                            item.get("name"),
                            item_id,
                        )
        elif et == "response.function_call_arguments.delta":
            item_id = event.get("item_id")
            delta = event.get("delta", "")
            if item_id and item_id in self._pending_tool_calls:
                self._pending_tool_calls[item_id]["argument_parts"].append(delta)
        elif et == "response.function_call_arguments.done":
            item_id = event.get("item_id")
            if item_id and item_id in self._pending_tool_calls:
                self._run_tool_in_background(self._execute_pending_tool_call(item_id))
        elif et == "response.output_item.done":
            item = event.get("item", {})
            item_id = item.get("id")
            if item_id and item_id in self._pending_tool_calls:
                self._run_tool_in_background(self._execute_pending_tool_call(item_id))
        elif et == "response.created":
            self._begin_response()
        elif et == "session.created":
            self.current_session = event.get("session")
            logger.info("Inworld session created")
        elif et == "session.updated":
            self.current_session = event.get("session")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Inworld session updated")
        elif et == "rate_limits.updated":
            self.current_rate_limits = event
        elif et == "response.done":
            response = event.get("response", {})
            status = response.get("status")
            if status == "failed":
                raise InworldRealtimeError(
                    f"Inworld realtime response failed: {response}"
                )
        elif et == "error":
            err = event.get("error", event)
            self._emit_error_event(
                InworldRealtimeError(json.dumps(err)),
                context="server_error",
            )
        elif et in (
            "conversation.item.created",
            "conversation.item.added",
            "conversation.item.done",
            "response.content_part.added",
            "response.content_part.done",
            "response.audio_transcript.delta",
            "response.output_audio_transcript.delta",
            "response.output_audio.done",
            "response.audio.done",
            "output_audio_buffer.started",
            "output_audio_buffer.stopped",
            "input_audio_buffer.speech_stopped",
            "input_audio_buffer.committed",
        ):
            pass
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Unrecognized Inworld Realtime event: %s %s", et, event)

    async def _handle_audio_output(self, pcm: PcmData) -> None:
        """Forward inbound audio from Inworld to the base-class event bus."""
        self._emit_audio_output_event(audio_data=pcm)

    async def _execute_pending_tool_call(self, item_id: str) -> None:
        """Execute a pending tool call after its arguments have fully streamed in."""
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

        logger.info("Executing tool call: %s with args: %s", name, arguments)

        tc, result, error = await self._run_one_tool(tool_call, timeout_s=30)

        if error:
            response_data: Dict[str, Any] = {"error": str(error)}
            logger.error("Tool call %s failed: %s", name, error)
        else:
            response_data = (
                {"result": result} if not isinstance(result, dict) else result
            )
            logger.info("Tool call %s succeeded", name)

        await self._send_tool_response(call_id, response_data)

    async def _send_tool_response(
        self, call_id: Optional[str], response_data: Dict[str, Any]
    ) -> None:
        """Send a tool-result item back to Inworld and trigger continuation."""
        if not call_id:
            logger.warning("Cannot send tool response without call_id")
            return

        response_str = self._sanitize_tool_output(response_data)

        await self.rtc._send_event(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": response_str,
                },
            }
        )
        logger.info("Sent tool response for call_id %s", call_id)
        await self.rtc._send_event({"type": "response.create"})

    def set_instructions(self, instructions: Instructions | str) -> None:
        super().set_instructions(instructions)
        self.realtime_session["instructions"] = self._instructions

    def _sanitize_tool_output(self, value: Any, max_chars: int = 60_000) -> str:
        """Serialize a tool result for transport back to Inworld."""
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    async def update_tools(self) -> None:
        """Re-send the registered tool list to Inworld.

        Use this if tools are registered after ``connect()`` is called;
        normally tools are included in the initial session config.
        """
        available_tools = self.get_available_functions()
        if not available_tools:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("No tools available to register with Inworld realtime")
            return

        tools_for_inworld = convert_tools_to_openai_format(
            available_tools, for_realtime=True
        )
        await self.rtc._send_event(
            {
                "type": "session.update",
                "session": {"tools": tools_for_inworld},
            }
        )
        logger.info(
            "Updated %d tools in Inworld realtime session", len(tools_for_inworld)
        )
