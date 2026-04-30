"""xAI Realtime API implementation for real-time AI audio communication using WebSocket.

The xAI SDK (verified through 1.11.0) provides AsyncClient for text/multimodal APIs
but ships no WebSocket wrapper for the realtime voice API. This implementation uses
the `websockets` library directly for the realtime connection while leveraging the
SDK's AsyncClient for ephemeral token generation and configuration.

See: https://docs.x.ai/developers/model-capabilities/audio/voice-agent
"""

import asyncio
import base64
import contextlib
import json
import logging
import os
from asyncio import CancelledError
from typing import Any, Optional
from urllib.parse import urlencode

import aiohttp
import websockets
from websockets.asyncio.client import ClientConnection
from xai_sdk import AsyncClient

from getstream.video.rtc.track_util import PcmData
from vision_agents.core.edge.types import Participant
from vision_agents.core.llm import realtime
from vision_agents.core.llm.events import LLMResponseChunkEvent
from vision_agents.core.llm.llm import LLMResponseEvent
from vision_agents.core.llm.llm_types import ToolSchema

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "grok-voice-think-fast-1.0"
DEFAULT_VOICE = "ara"
WEBSOCKET_URL = "wss://api.x.ai/v1/realtime"
EPHEMERAL_TOKEN_URL = "https://api.x.ai/v1/realtime/client_secrets"

# xAI's realtime model emits PCM at 24 kHz natively. Declaring a higher rate
# here causes the SDK-level PcmData to be tagged with the wrong rate, which
# then plays back at 2x speed and drains the buffer after ~1-2 seconds. The
# downstream audio pipeline resamples to the WebRTC track's rate as needed.
DEFAULT_SAMPLE_RATE = 24000


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

    Note: As of xai-sdk 1.11, the SDK still does not include a WebSocket wrapper
    for the realtime voice API, so this implementation uses the websockets library
    directly.

    Examples:

        from vision_agents.plugins import xai

        # Basic usage (web_search and x_search enabled by default)
        llm = xai.Realtime()
        await llm.connect()
        await llm.simple_response("Hello, how are you?")

        # With custom voice
        llm = xai.Realtime(voice="rex")

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

    - Audio format is PCM16 little-endian at 24 kHz (xAI's native model rate)
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
        vad_interrupt_response: bool = False,
        vad_threshold: float = 0.65,
        vad_prefix_padding_ms: int = 50,
        vad_silence_duration_ms: int = 200,
        interrupt_min_speech_ms: int = 100,
        web_search: bool = True,
        x_search: bool = True,
        x_search_allowed_handles: Optional[list[str]] = None,
        **kwargs,
    ) -> None:
        """Initialize xAI Realtime.

        Args:
            model: Model to use. Sent as the `model` query parameter on the
                   WebSocket URL. Defaults to "grok-voice-think-fast-1.0".
            voice: Voice to use for responses. Options: ara, rex, sal, eve, leo.
            api_key: Optional API key. Defaults to XAI_API_KEY environment variable.
            client: Optional AsyncClient instance. If not provided, one is created.
            turn_detection: Turn detection mode. Use "server_vad" for automatic
                           voice activity detection, or None for manual control.
            vad_interrupt_response: When True, the server auto-cancels the
                           assistant response on detected user speech. Defaults
                           to False because speaker-to-mic echo can otherwise
                           cancel the agent's own response mid-sentence.
            vad_threshold: Server VAD speech-detection confidence threshold
                           (0.0–1.0). Default 0.65 filters most brief noises
                           at the source so the debounce can stay short. Lower
                           (0.4–0.5) if the VAD is missing quiet speech; higher
                           (0.75+) if echo is triggering false speech_started.
            vad_prefix_padding_ms: Amount of audio (ms) the server VAD waits
                           before declaring speech started. Default 50ms is
                           snappy and works because vad_threshold is selective;
                           raise to 100–300 if the front of utterances is
                           being clipped from the model's input.
            vad_silence_duration_ms: How long the server VAD requires of
                           continuous silence before declaring the user's
                           turn complete (fires input_audio_buffer.committed).
                           Affects turn-end latency, not interrupt latency.
            interrupt_min_speech_ms: Debounce window applied locally before
                           treating detected user speech as an interruption.
                           Default 100ms — short enough to feel snappy, long
                           enough to filter most coughs and brief bursts that
                           clear the VAD threshold. Set to 0 for minimum
                           latency at the cost of cough-rejection; raise to
                           250+ if false-triggers are still cutting the
                           agent off. Total mouth-to-silence latency is
                           roughly VAD detection (~100ms) + this value +
                           ~200ms downstream pipeline tail.
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
        self.vad_interrupt_response = vad_interrupt_response
        self.vad_threshold = vad_threshold
        self.vad_prefix_padding_ms = vad_prefix_padding_ms
        self.vad_silence_duration_ms = vad_silence_duration_ms
        self.interrupt_min_speech_ms = interrupt_min_speech_ms
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
        # Tool tasks grouped by the response that requested them. We fire
        # exactly one response.create after all tool calls in a single
        # response have completed; firing one per call causes the model
        # to generate a separate spoken reply per parallel tool.
        self._response_tool_tasks: dict[str, list[asyncio.Task[None]]] = {}
        self._response_finalize_tasks: set[asyncio.Task[None]] = set()
        self._interrupt_timer: Optional[asyncio.Task[None]] = None
        self._interrupt_armed: bool = False
        self._active_response_id: Optional[str] = None

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
            aiohttp.ClientError: If token fetching fails.
        """
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

        uri = f"{WEBSOCKET_URL}?{urlencode({'model': self.model})}"
        try:
            self._ws = await self._exit_stack.enter_async_context(
                websockets.connect(
                    uri=uri,
                    additional_headers={"Authorization": f"Bearer {auth_token}"},
                )
            )
        except (OSError, websockets.WebSocketException, asyncio.TimeoutError) as e:
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
        """Send session configuration to xAI.

        Sends the OpenAI-realtime-compatible session.update payload that xAI
        expects. We mirror the shape used by the livekit xAI plugin (which
        extends `openai.realtime.RealtimeModel`) to stay aligned with what the
        server is known to accept.
        """
        config: dict[str, Any] = {
            "voice": self.voice,
            "modalities": ["text", "audio"],
            "input_audio_transcription": {},
            "audio": {
                "input": {"format": {"type": "audio/pcm", "rate": self.sample_rate}},
                "output": {"format": {"type": "audio/pcm", "rate": self.sample_rate}},
            },
        }

        if self._instructions:
            config["instructions"] = self._instructions

        if self.turn_detection:
            # Full ServerVad config. `interrupt_response=False` by default so
            # mic echo of the agent's own voice doesn't cancel the response
            # mid-sentence; toggle via `vad_interrupt_response=True` to opt in.
            config["turn_detection"] = {
                "type": self.turn_detection,
                "threshold": self.vad_threshold,
                "prefix_padding_ms": self.vad_prefix_padding_ms,
                "silence_duration_ms": self.vad_silence_duration_ms,
                "create_response": True,
                "interrupt_response": self.vad_interrupt_response,
            }
        else:
            config["turn_detection"] = None

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

        self._cancel_pending_interrupt()
        await self._await_pending_tools()
        if self._response_finalize_tasks:
            await asyncio.gather(*self._response_finalize_tasks, return_exceptions=True)

        if self._processing_task is not None:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except CancelledError:
                pass

        try:
            await self._exit_stack.aclose()
        except (OSError, websockets.WebSocketException, RuntimeError) as e:
            logger.warning(f"Error closing xAI session: {e}")

        self._ws = None

    async def simple_response(
        self,
        text: str,
        participant: Optional[Participant] = None,
    ) -> LLMResponseEvent[Any]:
        """
        Send a text message and request a response.

        Args:
            text: Text message to send.
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

            response_event = {"type": "response.create"}
            await self._send_event(response_event)

            return LLMResponseEvent(text="", original=None)
        except (ConnectionError, websockets.WebSocketException) as e:
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
        except (ConnectionError, websockets.WebSocketException) as e:
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
                except (websockets.WebSocketException, OSError, json.JSONDecodeError):
                    # Transient errors during message processing — keep the loop alive.
                    # Programming bugs (KeyError, AttributeError, ...) intentionally
                    # propagate so they aren't silently swallowed.
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

            if event_type in (
                "ping",
                "response.content_part.added",
                "response.content_part.done",
                "response.output_item.done",
            ):
                logger.debug("Received %s", event_type)

            elif event_type == "session.updated":
                logger.debug("xAI session configuration updated")

            elif event_type == "conversation.created":
                logger.debug("Conversation created: %s", data.get("conversation", {}))

            elif event_type == "conversation.item.added":
                self._handle_conversation_item_added(data)

            elif event_type == "conversation.item.input_audio_transcription.completed":
                # User speech transcription
                transcript = data.get("transcript", "")
                if transcript:
                    self._emit_user_speech_transcription(
                        text=transcript, mode="final", original=data
                    )

            elif event_type == "input_audio_buffer.speech_started":
                if self.interrupt_min_speech_ms > 0:
                    self._schedule_debounced_interrupt()
                else:
                    logger.info("🎙️  Speech detected — firing interrupt immediately")
                    await self._apply_interrupt()

            elif event_type == "input_audio_buffer.speech_stopped":
                # Don't cancel the debounce here. Server VAD fires this
                # after silence_duration_ms (~200ms), which is shorter than
                # the natural pauses between words/phrases — cancelling on
                # every dip would mean the debounce timer almost never
                # completes during real speech.
                logger.debug("Speech stopped detected")

            elif event_type == "input_audio_buffer.committed":
                # Server has decided the user's turn is truly complete.
                # Now it's safe to disarm the debounce — anything shorter
                # than this was a false alarm or a brief utterance.
                logger.debug("Audio buffer committed")
                self._cancel_pending_interrupt()

            elif event_type == "input_audio_buffer.cleared":
                logger.debug("Audio buffer cleared")

            elif event_type == "response.created":
                self._active_response_id = data.get("response", {}).get("id")
                self._begin_response()
                logger.debug(
                    "Response generation started: %s", self._active_response_id
                )

            elif event_type == "response.output_item.added":
                logger.debug("Response output item added")

            elif event_type == "response.output_audio_transcript.delta":
                delta = data.get("delta", "")
                if delta:
                    self._emit_agent_speech_transcription(
                        text=delta,
                        mode="delta",
                        original=data,
                    )

            elif event_type == "response.output_audio_transcript.done":
                self._emit_agent_speech_transcription(
                    text="",
                    mode="final",
                    original=data,
                )

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
                response_id = data.get("response", {}).get("id", "")
                if response_id == self._active_response_id:
                    self._active_response_id = None
                self._handle_response_done(data)
                self._maybe_finalize_tool_response(response_id)

            elif event_type in ("response.cancelled", "response.cancel"):
                # Server-initiated cancel — typically fired when server_vad
                # detects user speech and `interrupt_response=True`. Logging
                # at WARNING so we can see when the response is being cut off.
                response_id = data.get("response_id") or data.get("response", {}).get(
                    "id"
                )
                logger.warning(
                    "xAI cancelled response %s (reason: %s)",
                    response_id,
                    data.get("reason", "unspecified"),
                )
                if response_id == self._active_response_id:
                    self._active_response_id = None
                self._emit_audio_output_done_event(
                    response_id=response_id, interrupted=True
                )

            elif event_type == "rate_limits.updated":
                logger.debug("Rate limits updated: %s", data.get("rate_limits"))

            elif event_type == "response.function_call_arguments.done":
                # Function call from the model. Server-side tools (web_search,
                # x_search) are executed by the server and ignored here.
                self._handle_function_call_arguments_done(data)

            elif event_type == "error":
                error_info = data.get("error", {})
                error_msg = error_info.get("message", "Unknown error")
                logger.error(f"xAI realtime error: {error_msg}")
                self._emit_error_event(
                    error=Exception(error_msg),
                    context=f"xAI error: {error_info.get('type', 'unknown')}",
                )

            else:
                # Log unhandled events at INFO so diagnostics are visible
                # without needing DEBUG. Full payload still logged at DEBUG.
                logger.info("Unhandled xAI event type: %s", event_type)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Unhandled xAI event payload: %s", data)

    async def _apply_interrupt(self) -> None:
        """Stop the in-flight response: locally and (if known) server-side.

        Three things have to happen for the user to actually stop hearing
        the agent quickly:

        1. Local epoch bump so any audio events still in flight are dropped.
        2. ``audio_output_done(interrupted=True)`` so the agent's audio
           track flushes its buffer (which holds the model's lookahead).
        3. ``response.cancel`` to the server so it stops generating more
           audio chunks for this response. Without this, residual chunks
           from the old response can leak through after a new response
           starts (because ``_response_epoch`` is overwritten).
        """
        cancelled_id = self._active_response_id
        logger.info(
            "🛑 Applying interrupt (response_id=%s, vad_interrupt_response=%s)",
            cancelled_id,
            self.vad_interrupt_response,
        )
        await self.interrupt()
        if not self.vad_interrupt_response:
            self._emit_audio_output_done_event(interrupted=True)
        if cancelled_id is not None:
            try:
                await self._send_event(
                    {"type": "response.cancel", "response_id": cancelled_id}
                )
                logger.info("🛑 Sent response.cancel for %s", cancelled_id)
            except (ConnectionError, websockets.WebSocketException):
                logger.warning(
                    "Failed to send response.cancel for %s; connection issue",
                    cancelled_id,
                )

    def _schedule_debounced_interrupt(self) -> None:
        """Arm the debounce on the first speech_started of a turn.

        Subsequent speech_started events while already armed are ignored —
        we measure "speech is still going N ms after it started", not
        "N ms since the last burst", so re-arming would defeat the purpose.
        """
        if self._interrupt_armed:
            return
        self._interrupt_armed = True
        logger.info(
            "🎙️  Speech detected; arming interrupt timer (%dms)",
            self.interrupt_min_speech_ms,
        )
        self._interrupt_timer = asyncio.create_task(self._debounced_interrupt())

    def _cancel_pending_interrupt(self) -> None:
        """Disarm the debounce — the user's turn is over (committed)."""
        if self._interrupt_armed:
            logger.info("🎙️  Turn committed; disarming interrupt timer")
        self._interrupt_armed = False
        if self._interrupt_timer is not None and not self._interrupt_timer.done():
            self._interrupt_timer.cancel()
        self._interrupt_timer = None

    async def _debounced_interrupt(self) -> None:
        """Sleep the debounce window; interrupt unless disarmed in the meantime."""
        try:
            await asyncio.sleep(self.interrupt_min_speech_ms / 1000)
        except CancelledError:
            return
        if not self._interrupt_armed:
            return
        logger.info(
            "🎙️  Sustained speech past %dms — firing interrupt",
            self.interrupt_min_speech_ms,
        )
        await self._apply_interrupt()
        self._interrupt_armed = False
        self._interrupt_timer = None

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
        logger.debug("Response completed with status: %s", status)

        output = response.get("output", [])
        for item in output:
            if item.get("type") != "message":
                continue
            content = item.get("content", [])
            # If the response includes audio, the transcript was already
            # streamed in real-time via response.output_audio_transcript
            # events and written to the conversation by the agent's
            # on_realtime_agent_speech_transcription handler. Re-emitting
            # the text here would create a duplicate chat message.
            has_audio = any(cp.get("type") == "audio" for cp in content)
            if has_audio:
                continue
            for content_part in content:
                if content_part.get("type") == "text":
                    text = content_part.get("text", "")
                    if text:
                        event = LLMResponseChunkEvent(delta=text, plugin_name="xai")
                        self.events.send(event)

    def _handle_function_call_arguments_done(self, data: dict[str, Any]) -> None:
        """Schedule a local tool call for a finished function_call_arguments event.

        Server-side tools (e.g. x_search's x_keyword_search) emit the same
        event but are executed by the server; we must not reply with a
        function_call_output for those or the server treats it as an error
        and generates a second, apologetic response.

        Local tool tasks are tracked per response_id so that
        ``_maybe_finalize_tool_response`` can fire exactly one
        ``response.create`` after all parallel calls in a response complete.
        """
        function_name = data.get("name", "unknown")
        if self.function_registry.get_function(function_name) is None:
            logger.debug(
                "Ignoring server-side tool call %s (not in local registry)",
                function_name,
            )
            return

        response_id = data.get("response_id", "")
        task = asyncio.create_task(self._execute_tool_call(data))
        self._response_tool_tasks.setdefault(response_id, []).append(task)
        self._tool_tasks.add(task)
        task.add_done_callback(self._on_tool_task_done)

    async def _execute_tool_call(self, data: dict[str, Any]) -> None:
        """Run one local tool call and send its function_call_output.

        Continuation (``response.create``) is intentionally not sent here —
        ``_finalize_tool_response`` triggers it once after all tool calls
        for the response have completed.
        """
        function_name = data.get("name", "unknown")
        call_id = data.get("call_id", "")
        arguments_str = data.get("arguments", "{}")

        try:
            arguments = json.loads(arguments_str) if arguments_str else {}
        except json.JSONDecodeError:
            arguments = {}

        logger.debug(f'Calling function "{function_name}" with args "{arguments}"')

        tc, result, error = await self._run_one_tool(
            {
                "name": function_name,
                "arguments_json": arguments,
                "id": call_id,
            },
            timeout_s=30.0,
        )

        if error:
            output = json.dumps({"error": str(error)})
            logger.error(f"Function call {function_name} failed: {error}")
        else:
            output = result if isinstance(result, str) else json.dumps(result)

        function_output_event = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            },
        }
        await self._send_event(function_output_event)
        logger.debug(f'Function "{function_name}" output sent')

    def _maybe_finalize_tool_response(self, response_id: str) -> None:
        """If the completed response had local tool calls, schedule a single continuation."""
        if response_id not in self._response_tool_tasks:
            return
        finalize_task = asyncio.create_task(self._finalize_tool_response(response_id))
        self._response_finalize_tasks.add(finalize_task)
        finalize_task.add_done_callback(self._response_finalize_tasks.discard)

    async def _finalize_tool_response(self, response_id: str) -> None:
        """Await all tool tasks for ``response_id`` then trigger one continuation."""
        tasks = self._response_tool_tasks.pop(response_id, [])
        if not tasks:
            return
        await asyncio.gather(*tasks, return_exceptions=True)
        await self._send_event({"type": "response.create"})
        logger.debug(
            "Triggered continuation for response %s after %d tool call(s)",
            response_id,
            len(tasks),
        )

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
