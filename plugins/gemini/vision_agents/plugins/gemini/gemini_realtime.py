import asyncio
import contextlib
import copy
import logging
from asyncio import CancelledError
from typing import Any, Dict, List, Optional, cast

import aiortc
import av
from aiortc import VideoStreamTrack
from getstream.video.rtc.audio_track import AudioStreamTrack
from getstream.video.rtc.track_util import PcmData
from google import genai
from google.genai.live import AsyncSession
from google.genai.types import (
    AudioTranscriptionConfigDict,
    Blob,
    ContextWindowCompressionConfigDict,
    FunctionResponse,
    HttpOptions,
    LiveConnectConfigDict,
    LiveServerMessage,
    Modality,
    Part,
    PrebuiltVoiceConfigDict,
    RealtimeInputConfigDict,
    SessionResumptionConfigDict,
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
from vision_agents.core.llm.llm_types import NormalizedToolCallItem, ToolSchema
from vision_agents.core.processors import Processor
from vision_agents.core.utils.utils import frame_to_png_bytes
from vision_agents.core.utils.video_forwarder import VideoForwarder

logger = logging.getLogger(__name__)


"""
TODO:
- chat/transcription integration (trigger the right events when receiving transcriptions) - Tommaso
"""

DEFAULT_MODEL = "gemini-2.5-flash-native-audio-preview-09-2025"

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


# TODO: Move reconnection to the decorator
# TODO: self._session - make a property?
# TODO: Reconnect logic is broken: is_temporary always returns False
# TODO: Cleanup hasattr everywhere
# TODO: Cleanup tool calling, maybe move it


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
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.session_resumption_id: Optional[str] = None
        self.connected: bool = False

        http_options = http_options or HttpOptions(api_version="v1alpha")

        if client is not None:
            self._client = client
        elif api_key:
            self._client = genai.Client(api_key=api_key, http_options=http_options)
        else:
            self._client = genai.Client(http_options=http_options)

        self._config = copy.deepcopy(DEFAULT_CONFIG)
        # Merge custom config to the default config if provided
        if config:
            self._config.update(config)

        self._video_forwarder: Optional[VideoForwarder] = None
        self._real_session: Optional[AsyncSession] = None
        self._receive_task: Optional[asyncio.Task[Any]] = None
        self._exit_stack = contextlib.AsyncExitStack()

    @property
    def output_audio_track(self) -> AudioStreamTrack:
        return self._output_audio_track

    @property
    def _session(self):
        if not self._real_session:
            raise ValueError("The Gemini Session is not established yet")
        return self._real_session

    async def simple_response(
        self,
        text: str,
        processors: Optional[List[Processor]] = None,
        participant: Optional[Participant] = None,
    ):
        """
        Simple response standardizes how to send a text instruction to this LLM.

        Example:
            llm.simple_response("tell me a poem about Boulder")

        """
        logger.info("Simple response called with text: %s", text)
        try:
            await self._session.send_realtime_input(text=text)
        except Exception as e:
            # TODO: Move reconnection to the decorator
            # reconnect here in some cases
            logger.exception(e)
            is_temp = self._is_temporary_error(e)
            if is_temp:
                await self._reconnect()
            else:
                raise

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
        mime = "audio/pcm;rate=16000"
        blob = Blob(data=audio_bytes, mime_type=mime)

        await self._session.send_realtime_input(audio=blob)

    async def send_client_content(self, *args, **kwargs):
        """
        Don't use send client content, it can cause bugs when combined with send_realtime_input
        """
        await self._session.send_client_content(*args, **kwargs)

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

        if self._video_forwarder is not None and shared_forwarder is None:
            logger.info(
                "There's already a video forwarder already running, stopping previous one"
            )
            await self._stop_watching_video_track()

        self._video_forwarder = shared_forwarder or VideoForwarder(
            input_track=cast(VideoStreamTrack, track),
            max_buffer=5,
            fps=float(self.fps),
            name="gemini_forwarder",
        )

        # Add frame handler (starts automatically)
        self._video_forwarder.add_frame_handler(self._send_video_frame)
        logger.info(f"Started video forwarding with {self.fps} FPS")

    async def _stop_watching_video_track(self) -> None:
        if self._video_forwarder is not None:
            await self._video_forwarder.stop()
            self._video_forwarder = None
            logger.info("Stopped video forwarding")

    async def connect(self):
        """
        Connect to Gemini's websocket
        """
        logger.info("Connecting to Gemini live, config set to %s", self._config)
        self._real_session = await self._exit_stack.enter_async_context(
            self._client.aio.live.connect(  # type: ignore[arg-type]
                model=self.model, config=self._get_config_with_resumption()
            )
        )
        self.connected = True
        logger.info("Gemini live connected to session %s", self._session)

        # Start the receive loop task
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def close(self):
        self.connected = False

        if self._receive_task is not None:
            # TODO: Maybe we don't need to cancel it, we can just switch the connected flag and drain it
            self._receive_task.cancel()
            await self._receive_task

        try:
            await self._exit_stack.aclose()
        except Exception as e:
            logger.warning(f"Error closing session: {e}")
        self._real_session = None

    async def _reconnect(self):
        await self.connect()

    async def _receive_loop(self):
        """
        Main loop for receiving messages.

        Gemini's event system isn't ideal. It doesn't specify an event type with a clear structure.
        So you end up having to detect the type and reply as needed.
        Hopefully they will improve this in the future.
        """
        logger.info("Start processing events from Gemini Live API")
        try:
            while True:  # TODO: while Connected maybe?
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
                    is_interrupt = (
                        server_message
                        and server_message.server_content
                        and server_message.server_content.interrupted
                    )
                    is_turn_complete = (
                        server_message
                        and server_message.server_content
                        and server_message.server_content.turn_complete
                    )
                    is_generation_complete = (
                        server_message
                        and server_message.server_content
                        and server_message.server_content.generation_complete
                    )

                    if is_input_transcript:
                        if (
                            server_message.server_content
                            and server_message.server_content.input_transcription
                        ):
                            text = (
                                server_message.server_content.input_transcription.text
                            )
                            if text:
                                # TODO: should this be partial?
                                self._emit_user_speech_transcription(
                                    text=text, original=server_message
                                )
                    elif is_output_transcript:
                        if (
                            server_message.server_content
                            and server_message.server_content.output_transcription
                        ):
                            text = (
                                server_message.server_content.output_transcription.text
                            )
                            if text:
                                self._emit_agent_speech_transcription(
                                    text=text, original=server_message
                                )
                    elif is_interrupt:
                        if (
                            server_message.server_content
                            and server_message.server_content.interrupted
                        ):
                            logger.info(
                                "interrupted: %s",
                                server_message.server_content.interrupted,
                            )
                    elif is_response:
                        # Store the resumption id so we can resume a broken connection
                        if server_message.session_resumption_update:
                            update = server_message.session_resumption_update
                            if update.resumable and update.new_handle:
                                self.session_resumption_id = update.new_handle

                        if (
                            server_message.server_content
                            and server_message.server_content.model_turn
                        ):
                            parts = server_message.server_content.model_turn.parts

                            if parts:
                                for current_part in parts:
                                    typed_part: Part = current_part
                                    if typed_part.text:
                                        if typed_part.thought:
                                            logger.info(
                                                "Gemini thought %s", typed_part.text
                                            )
                                        else:
                                            logger.info("output: %s", typed_part.text)
                                            event = LLMResponseChunkEvent(
                                                delta=typed_part.text
                                            )
                                            self.events.send(event)
                                    elif typed_part.inline_data:
                                        # Emit audio output event
                                        pcm = PcmData.from_bytes(
                                            typed_part.inline_data.data, 24000
                                        )
                                        self._emit_audio_output_event(
                                            audio_data=pcm,
                                        )
                                    elif (
                                        hasattr(typed_part, "function_call")
                                        and typed_part.function_call
                                    ):
                                        # Handle function calls from Gemini Live
                                        logger.info(
                                            f"Received function call: {typed_part.function_call.name}"
                                        )
                                        await self._handle_function_call(
                                            typed_part.function_call
                                        )
                                    else:
                                        logger.debug(
                                            "Unrecognized part type: %s", typed_part
                                        )
                    elif is_turn_complete:
                        logger.info("is_turn_complete complete")
                    elif is_generation_complete:
                        logger.info("is_generation_complete complete")
                    elif server_message.tool_call:
                        # Handle tool calls from Gemini Live
                        logger.info(f"Received tool call: {server_message.tool_call}")
                        await self._handle_tool_call(server_message.tool_call)
                    else:
                        logger.warning(
                            "Unrecognized event structure for gemini %s", server_message
                        )
        except CancelledError:
            # TODO: Not sure about that
            logger.error("Stop async iteration exception")
            return

        except Exception as e:
            # reconnect here for some errors
            logger.exception("Error while processing events from Gemini Live API")
            is_temp = self._is_temporary_error(e)
            if is_temp:
                await self._reconnect()
            else:
                raise e
        finally:
            logger.info("Stop processing events from Gemini Live API")

    @staticmethod
    def _is_temporary_error(e: Exception):
        """
        Temporary errors should typically trigger a reconnect
        So if the websocket breaks this should return True and trigger a reconnect
        """

        should_reconnect = False
        return should_reconnect

    async def close(self):
        self.connected = False

        if hasattr(self, "_receive_task") and self._receive_task:
            self._receive_task.cancel()
            await self._receive_task

        if hasattr(self, "_session_context") and self._session_context:
            # Properly close the session using the context manager's __aexit__
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception as e:
                self.logger.warning(f"Error closing session: {e}")
            self._session_context = None
            self._session = None

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
        if self._video_forwarder is not None:
            await self._video_forwarder.remove_frame_handler(self._send_video_frame)
            self.logger.debug("Removed old video frame handler from previous forwarder")

        if shared_forwarder is not None:
            # Use the shared forwarder - just register as a consumer
            self._video_forwarder = shared_forwarder
            self.logger.info(
                f"🎥 Gemini subscribing to shared VideoForwarder at {self.fps} FPS"
            )
            self._video_forwarder.add_frame_handler(
                self._send_video_frame, fps=float(self.fps), name="gemini"
            )
        else:
            # Create our own VideoForwarder with the input track (legacy behavior)
            self._video_forwarder = VideoForwarder(
                track,  # type: ignore[arg-type]
                max_buffer=5,
                fps=float(self.fps),
                name="gemini_forwarder",
            )

            # Add frame handler (starts automatically)
            self._video_forwarder.add_frame_handler(self._send_video_frame)

            self.logger.info(f"Started video forwarding with {self.fps} FPS")

    async def _stop_watching_video_track(self) -> None:
        if self._video_forwarder is not None:
            await self._video_forwarder.stop()
            self._video_forwarder = None
            self.logger.info("Stopped video forwarding")

    async def _send_video_frame(self, frame: av.VideoFrame) -> None:
        """
        Send a video frame to Gemini using send_realtime_input
        """
        if not frame:
            return

        try:
            png_bytes = frame_to_png_bytes(frame)
            blob = Blob(data=png_bytes, mime_type="image/png")
            await self._session.send_realtime_input(media=blob)
        except Exception as e:
            logger.error(f"Error sending video frame: {e}")

    def _get_config_with_resumption(self) -> LiveConnectConfigDict:
        """
        _get_config_with_resumption adds the system instructions, session resumption, and tools
        """
        config = self._config.copy()
        # resume if we have a session resumption id/handle
        if self.session_resumption_id:
            resumption_config: SessionResumptionConfigDict = {
                "handle": self.session_resumption_id
            }  # type: ignore[typeddict-item]
            config["session_resumption"] = resumption_config  # type: ignore[typeddict-item]
        # set the instructions
        # TODO: potentially we can share the markdown as files/parts.. might do better TBD
        config["system_instruction"] = self._build_enhanced_instructions()

        # Add tools if available - Gemini Live uses similar format to regular Gemini
        tools_spec = self.get_available_functions()
        if tools_spec:
            conv_tools = self._convert_tools_to_provider_format(tools_spec)
            # Add tools to the live config
            # Note: The exact key name may need adjustment based on Gemini Live API documentation
            config["tools"] = conv_tools  # type: ignore[typeddict-item]
            logger.info(f"Added {len(tools_spec)} tools to Gemini Live config")
        else:
            logger.debug("No tools available - function calling will not work")

        return config

    def _convert_tools_to_provider_format(
        self, tools: List[ToolSchema]
    ) -> List[Dict[str, Any]]:
        """
        Convert ToolSchema objects to Gemini Live format.

        Args:
            tools: List of ToolSchema objects

        Returns:
            List of tools in Gemini Live format
        """
        function_declarations = []
        for tool in tools:
            function_declarations.append(
                {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool["parameters_schema"],
                }
            )

        # Return as dict with function_declarations (similar to regular Gemini format)
        return [{"function_declarations": function_declarations}]

    def _extract_tool_calls_from_response(
        self, response: Any
    ) -> List[NormalizedToolCallItem]:
        """
        Extract tool calls from Gemini Live response.

        Args:
            response: Gemini Live response object

        Returns:
            List of normalized tool call items
        """
        calls: List[NormalizedToolCallItem] = []

        try:
            # Check for function calls in the response
            if hasattr(response, "server_content") and response.server_content:
                if (
                    hasattr(response.server_content, "model_turn")
                    and response.server_content.model_turn
                ):
                    parts = response.server_content.model_turn.parts
                    for part in parts:
                        if hasattr(part, "function_call") and part.function_call:
                            call_item: NormalizedToolCallItem = {
                                "type": "tool_call",
                                "name": getattr(part.function_call, "name", "unknown"),
                                "arguments_json": getattr(
                                    part.function_call, "args", {}
                                ),
                            }
                            calls.append(call_item)
        except Exception as e:
            logger.debug(f"Error extracting tool calls from response: {e}")

        return calls

    async def _handle_tool_call(self, tool_call: Any) -> None:
        """
        Handle tool calls from Gemini Live.
        """
        try:
            if hasattr(tool_call, "function_calls") and tool_call.function_calls:
                for function_call in tool_call.function_calls:
                    await self._handle_function_call(function_call)
        except Exception as e:
            logger.error(f"Error handling tool call: {e}")

    async def _handle_function_call(self, function_call: Any) -> None:
        """
        Handle function calls from Gemini Live responses.

        Args:
            function_call: Function call object from Gemini Live
        """
        try:
            # Extract tool call details
            tool_call = {
                "name": getattr(function_call, "name", "unknown"),
                "arguments_json": getattr(function_call, "args", {}),
                "id": getattr(function_call, "id", None),
            }

            logger.info(
                f"Executing function call: {tool_call['name']} with args: {tool_call['arguments_json']}"
            )

            # Execute using existing tool execution infrastructure
            tc, result, error = await self._run_one_tool(tool_call, timeout_s=30)

            # Prepare response data
            if error:
                response_data = {"error": str(error)}
                logger.error(f"Function call {tool_call['name']} failed: {error}")
            else:
                # Ensure response is a dictionary for Gemini Live
                if not isinstance(result, dict):
                    response_data = {"result": result}
                else:
                    response_data = result
                logger.info(
                    f"Function call {tool_call['name']} succeeded: {response_data}"
                )

            # Send function response back to Gemini Live session
            call_id_val = tool_call.get("id")
            await self._send_function_response(
                str(tool_call["name"]),
                response_data,
                str(call_id_val) if call_id_val else None,
            )

        except Exception as e:
            logger.error(f"Error handling function call: {e}")
            # Send error response back
            await self._send_function_response(
                getattr(function_call, "name", "unknown"),
                {"error": str(e)},
                getattr(function_call, "id", None),
            )

    async def _send_function_response(
        self,
        function_name: str,
        response_data: Dict[str, Any],
        call_id: Optional[str] = None,
    ) -> None:
        """
        Send function response back to Gemini Live session.

        Args:
            function_name: Name of the function that was called
            response_data: Response data to send back
            call_id: Optional call ID for the function call
        """
        try:
            # Create function response part

            function_response = FunctionResponse(
                id=call_id,  # Use the call_id if provided
                name=function_name,
                response=response_data,
            )

            # Send the function response using the correct method
            # The Gemini Live API uses send_tool_response for function responses
            await self._session.send_tool_response(
                function_responses=[function_response]
            )
            logger.debug(f"Sent function response for {function_name}: {response_data}")

        except Exception:
            logger.exception(f"Error sending function response for {function_name}")
