import asyncio
import contextlib
import copy
import logging
from asyncio import CancelledError
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, cast

import aiortc
import av
import websockets
from aiortc import VideoStreamTrack
from getstream.video.rtc.track_util import PcmData
from google import genai
from google.genai.live import AsyncSession
from google.genai.types import (
    AudioTranscriptionConfigDict,
    Blob,
    ContextWindowCompressionConfigDict,
    FunctionCall,
    FunctionResponse,
    HttpOptions,
    LiveConnectConfigDict,
    LiveServerMessage,
    LiveServerToolCall,
    Modality,
    PrebuiltVoiceConfigDict,
    RealtimeInputConfigDict,
    SlidingWindowDict,
    SpeechConfigDict,
    TurnCoverage,
    VoiceConfigDict,
)
from vision_agents.core.edge.types import Participant
from vision_agents.core.llm import realtime
from vision_agents.core.llm.events import (
    LLMResponseChunkEvent,
)
from vision_agents.core.llm.llm import LLMResponseEvent
from vision_agents.core.llm.llm_types import ToolSchema
from vision_agents.core.processors import Processor
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.utils.video_utils import frame_to_png_bytes

from .file_search import FileSearchStore

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"

DEFAULT_CONFIG = LiveConnectConfigDict(
    response_modalities=[Modality.AUDIO],
    input_audio_transcription=AudioTranscriptionConfigDict(),
    output_audio_transcription=AudioTranscriptionConfigDict(),
    speech_config=SpeechConfigDict(
        voice_config=VoiceConfigDict(
            prebuilt_voice_config=PrebuiltVoiceConfigDict(voice_name="Leda")
        ),
        language_code="en-US",
    ),
    realtime_input_config=RealtimeInputConfigDict(
        turn_coverage=TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY
    ),
    enable_affective_dialog=False,
    context_window_compression=ContextWindowCompressionConfigDict(
        trigger_tokens=25600,
        sliding_window=SlidingWindowDict(target_tokens=12800),
    ),
)


def _should_reconnect(exc: Exception) -> bool:
    """
    Temporary errors should typically trigger a reconnect
    So if the websocket breaks this should return True and trigger a reconnect
    """
    # Gemini WS API returns code 1011 on session timeout
    reconnect_close_codes = [
        1011,  # Server-side exception or session timeout
        1012,  # Service restart
        1013,  # Try again later
        1014,  # Bad gateway
    ]
    if (
        isinstance(exc, websockets.ConnectionClosedError)
        and exc.rcvd
        and exc.rcvd.code in reconnect_close_codes
    ):
        return True
    return False


class GeminiRealtime(realtime.Realtime):
    """
    Gemini Realtime API implementation for real-time AI audio and video communication using Live API https://ai.google.dev/gemini-api/docs/live.

    Examples:

        config : LiveConnectConfigDict = {}
        model = "" # https://ai.google.dev/gemini-api/docs/live#audio-generation
        llm = Realtime(model="", config=config)
        # simple response
        llm.simple_response(text="Describe what you see and say hi")
        # native API call (forwards to gemini's send_realtime_input)
        llm.send_realtime_input()

        # Alternatively, you can also pass an existing client

        client = genai.Client()
        llm = Realtime(client=client)

    Development notes:

    - Audio data in the Live API is always raw, little-endian, 16-bit PCM.
    - Audio output always uses a sample rate of 24kHz.
    - Input audio is natively 16kHz, but the Live API will resample if needed
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        config: Optional[LiveConnectConfigDict] = None,
        http_options: Optional[HttpOptions] = None,
        client: Optional[genai.Client] = None,
        api_key: Optional[str] = None,
        file_search_store: Optional[FileSearchStore] = None,
        **kwargs,
    ) -> None:
        """
        Initialize Gemini Realtime.

        Args:
            model: Model to use for realtime.
            config: Optional LiveConnectConfigDict to customize behavior.
            http_options: Optional HTTP options.
            client: Optional Gemini client.
            api_key: Optional API key.
            file_search_store: Optional FileSearchStore for RAG functionality.
                See: https://ai.google.dev/gemini-api/docs/file-search
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self.model = model
        self.connected: bool = False
        self.file_search_store = file_search_store

        http_options = http_options or HttpOptions(api_version="v1alpha")

        if client is not None:
            self._client = client
        elif api_key:
            self._client = genai.Client(api_key=api_key, http_options=http_options)
        else:
            self._client = genai.Client(http_options=http_options)

        self._base_config = copy.deepcopy(DEFAULT_CONFIG)
        # Merge custom config to the default config if provided
        if config:
            self._base_config.update(config)

        self._session_resumption_id: Optional[str] = None
        self._video_forwarder: Optional[VideoForwarder] = None
        self._real_session: Optional[AsyncSession] = None
        self._processing_task: Optional[asyncio.Task] = None
        self._exit_stack = contextlib.AsyncExitStack()
        self._executor = ThreadPoolExecutor(max_workers=1)

    @property
    def _session(self):
        if not self._real_session:
            raise ValueError("The Gemini Session is not established yet")
        return self._real_session

    async def simple_response(
        self,
        text: str,
        processors: Optional[list[Processor]] = None,
        participant: Optional[Participant] = None,
    ) -> LLMResponseEvent[Any]:
        """
        Simple response standardizes how to send a text instruction to this LLM.

        Example:
            llm.simple_response("tell me a poem about Boulder")

        """
        try:
            await self._session.send_realtime_input(text=text)
            return LLMResponseEvent(text="", original=None)
        except Exception as e:
            # reconnect here in some cases
            if _should_reconnect(e):
                await self.connect()
            logger.exception("Failed to send realtime input to Gemini")
            return LLMResponseEvent(text="", original=None, exception=e)

    async def simple_audio_response(
        self, pcm: PcmData, participant: Optional[Participant] = None
    ):
        """
        Simple audio response standardizes how to send audio to the LLM

        Example:
            pcm : PcmData = PcmData()
            llm.simple_response(pcm)

        For more advanced use cases you can use the native send_realtime_input

        Args:
            pcm: PCM audio data to send
            participant: Optional participant information for the audio source
        """
        if not self.connected:
            return

        self._current_participant = participant

        # Build blob and send directly
        audio_bytes = pcm.resample(
            target_sample_rate=16000, target_channels=1
        ).samples.tobytes()
        blob = Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000")

        await self._session.send_realtime_input(audio=blob)

    async def watch_video_track(
        self,
        track: aiortc.mediastreams.MediaStreamTrack,
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        """
        Start sending video frames to Gemini using VideoForwarder.
        We follow the on_track from Stream. If video is turned on or off this gets forwarded.

        Args:
            track: Video track to watch
            shared_forwarder: Optional shared VideoForwarder to use instead of creating a new one
        """

        # This method can be called multiple times with different forwarders
        # Remove handler from old forwarder if it exists
        await self._stop_watching_video_track()
        if self._video_forwarder is not None:
            await self._video_forwarder.remove_frame_handler(self._send_video_frame)

        self._video_forwarder = shared_forwarder or VideoForwarder(
            input_track=cast(VideoStreamTrack, track),
            max_buffer=5,
            fps=float(self.fps),
            name="gemini_forwarder",
        )

        # Add frame handler (starts automatically)
        self._video_forwarder.add_frame_handler(self._send_video_frame)
        logger.info(f"Started video forwarding with {self.fps} FPS")

    async def _send_video_frame(self, frame: av.VideoFrame) -> None:
        """
        Send a video frame to Gemini using send_realtime_input

        Parameters:
            frame: Video frame to send.
        """
        loop = asyncio.get_running_loop()

        # Run frame conversion in a separate thread to avoid blocking the loop.
        png_bytes = await loop.run_in_executor(
            self._executor, frame_to_png_bytes, frame
        )

        blob = Blob(data=png_bytes, mime_type="image/png")
        try:
            await self._session.send_realtime_input(media=blob)
        except Exception:
            logger.exception("Failed to send a video frame to Gemini Live API")

    async def _stop_watching_video_track(self) -> None:
        if self._video_forwarder is not None:
            await self._video_forwarder.remove_frame_handler(self._send_video_frame)

    async def connect(self):
        """
        Connect to Gemini's websocket and start processing events.

        This method may be called multiple times in case of reconnects.
        Gemini Live API periodically closes websocket connection, and it must be re-established.
        """

        # Stop the processing task first in case we're reconnecting
        await self._stop_processing_task()

        logger.debug("Connecting to Gemini live, config set to %s", self._base_config)
        self._real_session = await self._exit_stack.enter_async_context(
            self._client.aio.live.connect(  # type: ignore[arg-type]
                model=self.model, config=self.get_config()
            )
        )
        self.connected = True
        logger.info("Gemini live connected to session %s", self._session)

        # Start the loop task
        await self._start_processing_task()

    async def close(self):
        """
        Close the LLM and clean up resources
        Returns:

        """
        self.connected = False

        await self._stop_watching_video_track()

        # Do not wait for threads to complete to avoid blocking the loop
        self._executor.shutdown(wait=False)

        if self._processing_task is not None:
            self._processing_task.cancel()
            await self._processing_task

        try:
            await self._exit_stack.aclose()
        except Exception as e:
            logger.warning(f"Error closing session: {e}")
        self._real_session = None

    def get_config(self) -> LiveConnectConfigDict:
        """
        Get Gemini Live config with additional runtime parameters like instructions, tools config,
        file search, and session resumption id.
        """
        config = self._base_config.copy()

        # Resume the session if we have a session resumption id
        if self._session_resumption_id:
            config["session_resumption"] = {"handle": self._session_resumption_id}

        # set the instructions
        config["system_instruction"] = self._instructions

        # Initialize tools list
        tools: list[dict] = []

        # Add function calling tools if available
        tools_spec = self.get_available_functions()
        if tools_spec:
            conv_tools = self._convert_tools_to_provider_format(tools_spec)
            tools.extend(conv_tools)
            logger.info(
                f"Adding {len(tools_spec)} function tools to Gemini Live config"
            )

        # Add file search tool if configured
        if self.file_search_store and self.file_search_store.is_created:
            file_search_config = self.file_search_store.get_tool_config()
            tools.append(file_search_config)
            logger.info("Adding file search tool to Gemini Live config")

        if tools:
            config["tools"] = tools  # type: ignore[typeddict-item]
        else:
            logger.debug("No tools available")

        return config

    async def _start_processing_task(self) -> None:
        self._processing_task = asyncio.create_task(self._processing_loop())

    async def _stop_processing_task(self) -> None:
        if self._processing_task is not None:
            self._processing_task.cancel()
            await self._processing_task

    async def _process_events(self) -> bool:
        """
        Process events from Gemini Live API.
        """
        async for response in self._session.receive():
            server_message: LiveServerMessage = response

            is_input_transcript = (
                server_message
                and server_message.server_content
                and server_message.server_content.input_transcription
            )
            is_output_transcript = (
                server_message
                and server_message.server_content
                and server_message.server_content.output_transcription
            )
            is_response = (
                server_message
                and server_message.server_content
                and server_message.server_content.model_turn
            )

            if is_input_transcript:
                if (
                    server_message.server_content
                    and server_message.server_content.input_transcription
                ):
                    text = server_message.server_content.input_transcription.text
                    if text:
                        self._emit_user_speech_transcription(
                            text=text, original=server_message
                        )
            elif is_output_transcript:
                if (
                    server_message.server_content
                    and server_message.server_content.output_transcription
                ):
                    text = server_message.server_content.output_transcription.text
                    if text:
                        self._emit_agent_speech_transcription(
                            text=text, original=server_message
                        )
            elif is_response:
                # Store the resumption id so we can resume a broken connection
                if server_message.session_resumption_update:
                    update = server_message.session_resumption_update
                    if update.resumable and update.new_handle:
                        self._session_resumption_id = update.new_handle

                if (
                    server_message.server_content
                    and server_message.server_content.model_turn
                ):
                    parts = server_message.server_content.model_turn.parts or []
                    for part in parts:
                        if part.text and not part.thought:
                            # Emit partial LLM response event
                            event = LLMResponseChunkEvent(delta=part.text)
                            self.events.send(event)
                        elif part.inline_data:
                            # Emit audio output event
                            pcm = PcmData.from_bytes(part.inline_data.data, 24000)
                            self._emit_audio_output_event(audio_data=pcm)
                        elif part.function_call:
                            # Handle function calls from Gemini Live
                            await self._handle_function_call(part.function_call)

            elif server_message.tool_call:
                # Handle tool calls from Gemini Live
                await self._handle_tool_calls(server_message.tool_call)
            else:
                logger.debug(
                    "Unrecognized event structure from Gemini %s", server_message
                )

        return False

    async def _processing_loop(self):
        """
        Start the loop for receiving messages.

        It also reconnects the underlying session if it's closed.
        """
        logger.debug("Start processing events from Gemini Live API")
        try:
            while True:
                try:
                    await self._process_events()
                except websockets.ConnectionClosedError as e:
                    if not _should_reconnect(e):
                        raise e
                    # Reconnect here for some errors
                    await self.connect()
                except Exception:
                    logger.exception(
                        "Error while processing events from Gemini Live API"
                    )

        except CancelledError:
            logger.debug("Processing loop has been cancelled")

    def _convert_tools_to_provider_format(
        self, tools: list[ToolSchema]
    ) -> list[dict[str, Any]]:
        """
        Convert ToolSchema objects to Gemini Live format.

        Args:
            tools: List of ToolSchema objects

        Returns:
            List of tools in Gemini Live format
        """
        function_declarations = [
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool["parameters_schema"],
            }
            for tool in tools
        ]

        # Return as dict with function_declarations (similar to regular Gemini format)
        return [{"function_declarations": function_declarations}]

    async def _handle_tool_calls(self, tool_call: LiveServerToolCall) -> None:
        """
        Handle tool calls from Gemini Live.
        """
        function_calls = tool_call.function_calls or []
        for function_call in function_calls:
            await self._handle_function_call(function_call)

    async def _handle_function_call(
        self, function_call: FunctionCall, timeout: float = 30.0
    ) -> None:
        """
        Handle function calls from Gemini Live responses.

        Args:
            function_call: Function call object from Gemini Live
            timeout: Function call timeout in seconds
        """

        # Extract tool call details
        function_name = function_call.name or "unknown"
        function_args = function_call.args or {}
        call_id = function_call.id

        logger.debug(f'Calling function "{function_name}" with args "{function_args}"')
        # Execute using existing tool execution infrastructure
        tc, result, error = await self._run_one_tool(
            {
                "name": function_name,
                "arguments_json": function_args,
                "id": call_id,
            },
            timeout_s=timeout,
        )

        # Prepare response data
        if error:
            response_data = {"error": str(error)}
            logger.error(f"Function call {function_name} failed: {error}")
        else:
            # Ensure response is a dictionary for Gemini Live
            response_data = result if isinstance(result, dict) else {"result": result}

        # Send function response back to Gemini Live session
        function_response = FunctionResponse(
            id=call_id, name=function_name, response=response_data
        )
        # Send the function response back to the Gemini Live API
        logger.debug(f'Send a function response for "{function_name}": {response_data}')
        try:
            await self._session.send_tool_response(
                function_responses=[function_response]
            )
        except Exception:
            logger.exception(
                f'Failed to send a response for function "{function_name}" back to Gemini'
            )
