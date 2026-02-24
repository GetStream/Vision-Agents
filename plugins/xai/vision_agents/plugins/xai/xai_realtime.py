"""
xAI Realtime API implementation for real-time AI audio communication using WebSocket.

The xAI SDK (as of 1.5.0) provides AsyncClient for text/multimodal APIs, but does not
include a WebSocket wrapper for the realtime voice API. This implementation uses the
websockets library directly for the realtime connection while leveraging the SDK's
AsyncClient for ephemeral token generation and configuration.

See: https://docs.x.ai/docs/guides/voice/agent
"""

import asyncio
import base64
import contextlib
import json
import logging
import os
from asyncio import CancelledError
from typing import Any, Optional

import websockets
from websockets.asyncio.client import ClientConnection
from xai_sdk import AsyncClient

from getstream.video.rtc.track_util import PcmData
from vision_agents.core.edge.types import Participant
from vision_agents.core.llm import realtime
from vision_agents.core.llm.events import LLMResponseChunkEvent
from vision_agents.core.llm.llm import LLMResponseEvent
from vision_agents.core.llm.llm_types import ToolSchema
from vision_agents.core.processors import Processor

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "grok-3-fast"
DEFAULT_VOICE = "Ara"
WEBSOCKET_URL = "wss://api.x.ai/v1/realtime"
EPHEMERAL_TOKEN_URL = "https://api.x.ai/v1/realtime/client_secrets"

# Default sample rate, matching webrtc here
DEFAULT_SAMPLE_RATE = 48000


def _should_reconnect(exc: Exception) -> bool:
    """Determine if the connection should be reconnected based on the exception."""
    reconnect_close_codes = [
        1011,  # Server-side exception or session timeout
        1012,  # Service restart
        1013,  # Try again later
        1014,  # Bad gateway
    ]
    if isinstance(exc, websockets.ConnectionClosedError):
        if exc.rcvd and exc.rcvd.code in reconnect_close_codes:
            return True
    return False


class XAIRealtime(realtime.Realtime):
    """
    xAI Realtime API implementation for real-time voice conversations.

    Uses WebSocket connection to xAI's realtime endpoint for bidirectional
    audio streaming with voice AI capabilities. The SDK's AsyncClient is used
    for configuration and potential ephemeral token generation.

    Note: The xAI SDK (1.5.0+) does not yet include a WebSocket wrapper for the
    realtime voice API, so this implementation uses the websockets library directly.

    Examples:

        from vision_agents.plugins import xai

        # Basic usage (web_search and x_search enabled by default)
        llm = xai.Realtime()
        await llm.connect()
        await llm.simple_response("Hello, how are you?")

        # With custom voice
        llm = xai.Realtime(voice="Rex")

        # Disable web search and X search
        llm = xai.Realtime(web_search=False, x_search=False)

        # Restrict X search to specific handles
        llm = xai.Realtime(x_search_allowed_handles=["elonmusk", "xai"])

        # With API key
        llm = xai.Realtime(api_key="your-api-key")

        # With existing AsyncClient
        client = AsyncClient(api_key="your-api-key")
        llm = xai.Realtime(client=client)

    Development notes:

    - Audio format is PCM16 little-endian at 48kHz
    - Supports server-side VAD (voice activity detection) by default
    - Web search and X search are enabled by default
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        api_key: Optional[str] = None,
        client: Optional[AsyncClient] = None,
        turn_detection: Optional[str] = "server_vad",
        web_search: bool = True,
        x_search: bool = True,
        x_search_allowed_handles: Optional[list[str]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize xAI Realtime.

        Args:
            model: Model to use (currently informational, not sent to API).
            voice: Voice to use for responses. Options: Ara, Rex, Sal, Eve, Leo.
            api_key: Optional API key. Defaults to XAI_API_KEY environment variable.
            client: Optional AsyncClient instance. If not provided, one is created.
            turn_detection: Turn detection mode. Use "server_vad" for automatic
                           voice activity detection, or None for manual control.
            web_search: Enable web search tool. Defaults to True.
            x_search: Enable X (Twitter) search tool. Defaults to True.
            x_search_allowed_handles: Optional list of X handles to restrict search to.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self.model = model
        self.voice = voice
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.turn_detection = turn_detection
        self.web_search = web_search
        self.x_search = x_search
        self.x_search_allowed_handles = x_search_allowed_handles
        self.provider_name = "xai"

        # Initialize API key and client
        self._api_key = api_key or os.environ.get("XAI_API_KEY")
        if not self._api_key and client is None:
            raise ValueError(
                "XAI API key is required. Set XAI_API_KEY environment variable, "
                "pass api_key parameter, or provide an AsyncClient."
            )

        # Use provided client or create one
        if client is not None:
            self._client = client
        elif self._api_key:
            self._client = AsyncClient(api_key=self._api_key)
        else:
            self._client = AsyncClient()

        self._ws: Optional[ClientConnection] = None
        self._processing_task: Optional[asyncio.Task] = None
        self._exit_stack = contextlib.AsyncExitStack()
        self._ephemeral_token: Optional[str] = None

    async def get_ephemeral_token(self, expires_seconds: int = 300) -> str:
        """
        Fetch an ephemeral token for client-side authentication.

        Ephemeral tokens are recommended for client-side applications where
        exposing the API key would be a security risk.

        Args:
            expires_seconds: Token expiration time in seconds (default: 300).

        Returns:
            The ephemeral token string.

        Raises:
            Exception: If token fetching fails.
        """
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.post(
                EPHEMERAL_TOKEN_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={"expires_after": {"seconds": expires_seconds}},
            ) as response:
                response.raise_for_status()
                data = await response.json()
                token = data.get("client_secret", {}).get("value")
                if not token:
                    raise ValueError("No token in response")
                self._ephemeral_token = token
                logger.debug(
                    "Fetched ephemeral token (expires in %d seconds)", expires_seconds
                )
                return token

    async def connect(self, use_ephemeral_token: bool = False):
        """
        Connect to xAI's WebSocket endpoint and start processing events.

        Args:
            use_ephemeral_token: If True, fetch an ephemeral token for authentication
                                 instead of using the API key directly. Recommended
                                 for client-side applications.

        This method may be called multiple times in case of reconnects.
        """
        # Stop the processing task first in case we're reconnecting
        await self._stop_processing_task()

        logger.debug("Connecting to xAI realtime API")

        # Get authentication token
        if use_ephemeral_token:
            auth_token = await self.get_ephemeral_token()
        else:
            # _api_key is guaranteed to be set if client was not provided (checked in __init__)
            assert self._api_key is not None
            auth_token = self._api_key

        try:
            self._ws = await self._exit_stack.enter_async_context(
                websockets.connect(
                    uri=WEBSOCKET_URL,
                    additional_headers={"Authorization": f"Bearer {auth_token}"},
                )
            )
        except Exception as e:
            logger.error(f"Failed to connect to xAI realtime: {e}")
            logger.error("Check that XAI_API_KEY is valid and has realtime API access")
            raise

        # Configure the session
        await self._configure_session()

        self.connected = True
        self._emit_connected_event(
            session_config={
                "voice": self.voice,
                "turn_detection": self.turn_detection,
            },
            capabilities=["audio", "text", "function_calling"],
        )
        logger.info("xAI realtime connected")

        # Start the event processing loop
        await self._start_processing_task()

    async def _configure_session(self) -> None:
        """Send session configuration to xAI."""
        config: dict[str, Any] = {
            "voice": self.voice,
            "audio": {
                "input": {"format": {"type": "audio/pcm", "rate": self.sample_rate}},
                "output": {"format": {"type": "audio/pcm", "rate": self.sample_rate}},
            },
        }

        if self._instructions:
            config["instructions"] = self._instructions

        if self.turn_detection:
            config["turn_detection"] = {"type": self.turn_detection}
        else:
            config["turn_detection"] = {"type": None}

        # Build tools list
        tools: list[dict[str, Any]] = []

        # Add web search tool if enabled
        if self.web_search:
            tools.append({"type": "web_search"})

        # Add X search tool if enabled
        if self.x_search:
            x_search_tool: dict[str, Any] = {"type": "x_search"}
            if self.x_search_allowed_handles:
                x_search_tool["allowed_x_handles"] = self.x_search_allowed_handles
            tools.append(x_search_tool)

        # Add user-defined function tools
        function_tools = self._get_tools_for_provider()
        if function_tools:
            tools.extend(function_tools)

        if tools:
            config["tools"] = tools

        session_update = {"type": "session.update", "session": config}
        await self._send_event(session_update)

    async def _send_event(self, event: dict[str, Any]) -> None:
        """Send an event to the WebSocket."""
        if not self._ws:
            raise ConnectionError("WebSocket is not connected")
        await self._ws.send(json.dumps(event))

    async def close(self):
        """Close the connection and clean up resources."""
        self.connected = False
        self._emit_disconnected_event(reason="close requested", was_clean=True)

        if self._processing_task is not None:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except CancelledError:
                pass

        try:
            await self._exit_stack.aclose()
        except Exception as e:
            logger.warning(f"Error closing xAI session: {e}")

        self._ws = None

    async def simple_response(
        self,
        text: str,
        processors: Optional[list[Processor]] = None,
        participant: Optional[Participant] = None,
    ) -> LLMResponseEvent[Any]:
        """
        Send a text message and request a response.

        Args:
            text: Text message to send.
            processors: Optional list of processors (unused).
            participant: Optional participant information.

        Returns:
            LLMResponseEvent with empty text (actual response comes via events).
        """
        try:
            # Create a conversation item with the text
            create_event = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
            await self._send_event(create_event)

            # Request a response
            response_event = {
                "type": "response.create",
                "response": {"modalities": ["text", "audio"]},
            }
            await self._send_event(response_event)

            return LLMResponseEvent(text="", original=None)
        except Exception as e:
            if _should_reconnect(e):
                await self.connect()
            logger.exception("Failed to send message to xAI realtime")
            return LLMResponseEvent(text="", original=None, exception=e)

    async def simple_audio_response(
        self, pcm: PcmData, participant: Optional[Participant] = None
    ):
        """
        Send audio data to xAI realtime.

        Args:
            pcm: PCM audio data to send.
            participant: Optional participant information.
        """
        if not self.connected:
            return

        self._current_participant = participant

        # Resample audio to target sample rate if needed
        resampled = pcm.resample(target_sample_rate=self.sample_rate, target_channels=1)
        audio_bytes = resampled.samples.tobytes()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        # Append audio to buffer
        append_event = {"type": "input_audio_buffer.append", "audio": audio_base64}
        try:
            await self._send_event(append_event)
        except Exception as e:
            if _should_reconnect(e):
                await self.connect()
            logger.exception("Failed to send audio to xAI realtime")

    async def commit_audio_buffer(self) -> None:
        """
        Commit the audio buffer to create a user message.

        Only needed when turn_detection is None (manual mode).
        With server_vad, the server automatically commits based on speech detection.
        """
        commit_event = {"type": "input_audio_buffer.commit"}
        await self._send_event(commit_event)

    async def clear_audio_buffer(self) -> None:
        """Clear the input audio buffer."""
        clear_event = {"type": "input_audio_buffer.clear"}
        await self._send_event(clear_event)

    async def _start_processing_task(self) -> None:
        """Start the event processing background task."""
        self._processing_task = asyncio.create_task(self._processing_loop())

    async def _stop_processing_task(self) -> None:
        """Stop the event processing background task."""
        if self._processing_task is not None:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except CancelledError:
                pass

    async def _processing_loop(self):
        """Main loop for receiving and processing WebSocket events."""
        logger.debug("Starting xAI realtime event processing loop")
        try:
            while True:
                try:
                    await self._process_events()
                except websockets.ConnectionClosedError as e:
                    if not _should_reconnect(e):
                        raise e
                    logger.warning(
                        f"xAI WebSocket closed with code {e.rcvd.code if e.rcvd else 'unknown'}, reconnecting..."
                    )
                    await self.connect()
                except Exception:
                    logger.exception("Error while processing xAI realtime events")
        except CancelledError:
            logger.debug("xAI realtime processing loop cancelled")

    async def _process_events(self) -> None:
        """Process events from the xAI WebSocket connection."""
        if not self._ws:
            raise ConnectionError("WebSocket is not connected")

        async for message in self._ws:
            data = json.loads(message)
            event_type = data.get("type", "")

            logger.debug(f"Received xAI event: {event_type}")

            if event_type == "session.updated":
                logger.debug("xAI session configuration updated")

            elif event_type == "conversation.created":
                logger.debug("Conversation created: %s", data.get("conversation", {}))

            elif event_type == "conversation.item.added":
                self._handle_conversation_item_added(data)

            elif event_type == "conversation.item.input_audio_transcription.completed":
                # User speech transcription
                transcript = data.get("transcript", "")
                if transcript:
                    self._emit_user_speech_transcription(text=transcript, original=data)

            elif event_type == "input_audio_buffer.speech_started":
                logger.debug("Speech started detected")

            elif event_type == "input_audio_buffer.speech_stopped":
                logger.debug("Speech stopped detected")

            elif event_type == "input_audio_buffer.committed":
                logger.debug("Audio buffer committed")

            elif event_type == "input_audio_buffer.cleared":
                logger.debug("Audio buffer cleared")

            elif event_type == "response.created":
                logger.debug("Response generation started")

            elif event_type == "response.output_item.added":
                logger.debug("Response output item added")

            elif event_type == "response.output_audio_transcript.delta":
                # Agent speech transcript delta
                delta = data.get("delta", "")
                if delta:
                    self._emit_agent_speech_transcription(text=delta, original=data)

            elif event_type == "response.output_audio_transcript.done":
                logger.debug("Agent transcript complete")

            elif event_type == "response.output_audio.delta":
                # Audio output from the model
                audio_base64 = data.get("delta", "")
                if audio_base64:
                    audio_bytes = base64.b64decode(audio_base64)
                    pcm = PcmData.from_bytes(audio_bytes, self.sample_rate)
                    self._emit_audio_output_event(
                        audio_data=pcm, response_id=data.get("response_id")
                    )

            elif event_type == "response.output_audio.done":
                logger.debug("Audio output complete")
                self._emit_audio_output_done_event(response_id=data.get("response_id"))

            elif event_type == "response.done":
                self._handle_response_done(data)

            elif event_type == "response.function_call_arguments.done":
                # Function call from the model
                await self._handle_function_call(data)

            elif event_type == "error":
                error_info = data.get("error", {})
                error_msg = error_info.get("message", "Unknown error")
                logger.error(f"xAI realtime error: {error_msg}")
                self._emit_error_event(
                    error=Exception(error_msg),
                    context=f"xAI error: {error_info.get('type', 'unknown')}",
                )

            else:
                logger.debug(f"Unhandled xAI event type: {event_type}")

    def _handle_conversation_item_added(self, data: dict[str, Any]) -> None:
        """Handle conversation.item.added event."""
        item = data.get("item", {})
        item_id = item.get("id")
        item_type = item.get("type")
        role = item.get("role")
        content = item.get("content", [])

        self._emit_conversation_item_event(
            item_id=item_id,
            item_type=item_type,
            status=item.get("status", "completed"),
            role=role,
            content=content,
        )

    def _handle_response_done(self, data: dict[str, Any]) -> None:
        """Handle response.done event."""
        response = data.get("response", {})
        status = response.get("status", "completed")
        logger.debug(f"Response completed with status: {status}")

        # Emit text delta if there's any text content
        output = response.get("output", [])
        for item in output:
            if item.get("type") == "message":
                for content_part in item.get("content", []):
                    if content_part.get("type") == "text":
                        text = content_part.get("text", "")
                        if text:
                            event = LLMResponseChunkEvent(delta=text)
                            self.events.send(event)

    async def _handle_function_call(self, data: dict[str, Any]) -> None:
        """Handle function calls from xAI realtime."""
        function_name = data.get("name", "unknown")
        call_id = data.get("call_id", "")
        arguments_str = data.get("arguments", "{}")

        try:
            arguments = json.loads(arguments_str) if arguments_str else {}
        except json.JSONDecodeError:
            arguments = {}

        logger.debug(f'Calling function "{function_name}" with args "{arguments}"')

        # Execute using existing tool execution infrastructure
        tc, result, error = await self._run_one_tool(
            {
                "name": function_name,
                "arguments_json": arguments,
                "id": call_id,
            },
            timeout_s=30.0,
        )

        # Prepare output
        if error:
            output = json.dumps({"error": str(error)})
            logger.error(f"Function call {function_name} failed: {error}")
        else:
            output = result if isinstance(result, str) else json.dumps(result)

        # Send function result back to xAI
        function_output_event = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            },
        }
        await self._send_event(function_output_event)

        # Request continuation
        response_event = {"type": "response.create"}
        await self._send_event(response_event)

        logger.debug(f'Function "{function_name}" response sent')

    def _convert_tools_to_provider_format(
        self, tools: list[ToolSchema]
    ) -> list[dict[str, Any]]:
        """
        Convert ToolSchema objects to xAI realtime format.

        Args:
            tools: List of ToolSchema objects.

        Returns:
            List of tools in xAI format.
        """
        result = []
        for tool in tools:
            params = tool.get("parameters_schema") or tool.get("parameters") or {}
            if not isinstance(params, dict):
                params = {}
            params.setdefault("type", "object")
            params.setdefault("properties", {})

            result.append(
                {
                    "type": "function",
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": params,
                }
            )
        return result

    async def watch_video_track(self, track, shared_forwarder=None) -> None:
        """
        xAI realtime currently does not support video input.

        This method is a no-op for API compatibility.
        """
        logger.warning(
            "xAI realtime does not support video input - ignoring video track"
        )

    async def stop_watching_video_track(self) -> None:
        """Stop watching video track (no-op for xAI)."""
        pass
