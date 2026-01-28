import asyncio
import logging
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Literal, Optional, cast

import av
from aiortc.mediastreams import MediaStreamTrack, VideoStreamTrack
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant
from google.genai import types
from google.genai.client import AsyncClient, Client
from google.genai.types import GenerateContentConfig, MediaResolution, ThinkingLevel

import json
import re

from vision_agents.core.llm.events import (
    LLMRequestStartedEvent,
    LLMResponseChunkEvent,
    LLMResponseCompletedEvent,
    VLMAnnotationEvent,
    VLMInferenceStartEvent,
    VLMInferenceCompletedEvent,
    VLMErrorEvent,
)
from vision_agents.core.llm.llm import LLMResponseEvent, VideoLLM
from vision_agents.core.llm.llm_types import NormalizedToolCallItem, ToolSchema
from vision_agents.core.processors import Processor
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.utils.video_utils import frame_to_jpeg_bytes, frame_to_png_bytes

from . import events
from .gemini_llm import DEFAULT_MODEL
from .tools import GeminiTool

logger = logging.getLogger(__name__)

PLUGIN_NAME = "gemini_vlm"


class GeminiVLM(VideoLLM):
    """
    Gemini Video Language Model (VLM) implementation for video understanding.

    This plugin extends Gemini's LLM capabilities with video processing, supporting
    two modes of operation:

    **Mode 1: Automatic Frame Buffering (fps > 0)**
    Continuously buffers video frames and includes them with every LLM request.
    Best for scenarios requiring constant visual context.

    **Mode 2: On-Demand Frame Capture (fps = 0)**
    Only captures and sends frames when explicitly requested via function calls.
    More efficient when you only need vision occasionally. The LLM can call
    `capture_frame()` or `analyze_video()` functions to access the current frame.

    Features:
        - Video understanding: Two flexible modes for frame handling
        - Streaming responses: Real-time chunk events with low latency
        - Frame buffering: Configurable FPS and buffer duration
        - On-demand capture: Hook-based frame capture via function calls
        - Function calling: Full support for Gemini function calling
        - Tool execution: Multi-hop tool calling with deduplication
        - Gemini 3 features: Thinking levels, media resolution, code execution

    Examples:

        from vision_agents.plugins import gemini

        # Mode 1: Automatic buffering - always includes video context
        llm = gemini.VLM(
            model="gemini-2.0-flash-exp",
            fps=1,  # Buffer 1 frame per second
            frame_buffer_seconds=10
        )

        # Mode 2: On-demand capture - only when requested
        llm = gemini.VLM(
            model="gemini-2.0-flash-exp",
            fps=0,  # Disable automatic buffering
            enable_vision_tools=True  # Enable capture_frame() function
        )
        # Now the LLM can call capture_frame() when user asks about video

        # Advanced: Hybrid approach with custom tools
        llm = gemini.VLM(
            model="gemini-3-pro-preview",
            fps=0,
            enable_vision_tools=True,
            thinking_level=ThinkingLevel.HIGH,
            media_resolution=MediaResolution.MEDIA_RESOLUTION_HIGH
        )
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        client: Optional[AsyncClient] = None,
        fps: int = 1,
        frame_buffer_seconds: int = 10,
        frame_format: Literal["png", "jpeg"] = "jpeg",
        frame_width: int = 800,
        frame_height: int = 600,
        max_workers: int = 4,
        enable_vision_tools: bool = False,
        enable_structured_annotations: bool = False,
        thinking_level: Optional[ThinkingLevel] = None,
        media_resolution: Optional[MediaResolution] = None,
        config: Optional[GenerateContentConfig] = None,
        tools: Optional[List[GeminiTool]] = None,
        **kwargs,
    ):
        """
        Initialize the GeminiVLM class.

        Args:
            model: The Gemini model to use. Defaults to gemini-3-pro-preview.
            api_key: Optional API key. By default loads from GOOGLE_API_KEY.
            client: Optional Gemini client. By default creates a new client object.
            fps: Frames per second to process. Set to 0 for on-demand mode. Default: 1.
                - fps > 0: Automatic buffering mode - continuously captures frames
                - fps = 0: On-demand mode - only captures when function is called
            frame_buffer_seconds: Seconds to buffer (only used if fps > 0). Default: 10.
                Total buffer size = fps * frame_buffer_seconds.
            frame_format: Format for video frames ("png" or "jpeg"). Default: "jpeg".
            frame_width: The width of the video frame to send. Default: 800.
            frame_height: The height of the video frame to send. Default: 600.
            max_workers: Max worker threads for frame conversion. Default: 4.
            enable_vision_tools: If True, automatically registers vision functions
                that the LLM can call to capture frames on-demand. Useful with fps=0.
                Registers: capture_frame() and analyze_video(). Default: False.
            enable_structured_annotations: If True, instructs the LLM to return visual
                annotations as structured JSON when asked to mark, circle, or highlight
                objects. The annotations will be drawn on video by AnnotationProcessor.
                Format: [{"box_2d": [x1,y1,x2,y2], "label": "..."}]. Default: False.
            thinking_level: Optional thinking level for Gemini 3. Use ThinkingLevel.LOW
                or ThinkingLevel.HIGH. Defaults to "high" for Gemini 3 Pro.
            media_resolution: Optional media resolution for multimodal processing.
                Use MEDIA_RESOLUTION_LOW, MEDIA_RESOLUTION_MEDIUM, or
                MEDIA_RESOLUTION_HIGH. Recommended: "high" for images/video.
            config: Optional GenerateContentConfig to use as base. Any kwargs will be
                passed to GenerateContentConfig constructor if config is not provided.
            tools: Optional list of Gemini built-in tools. Available tools:
                - tools.FileSearch(store): RAG over your documents
                - tools.GoogleSearch(): Ground responses with web data
                - tools.CodeExecution(): Run Python code
                - tools.URLContext(): Read specific web pages
                - tools.GoogleMaps(): Location-aware queries (Preview)
                - tools.ComputerUse(): Browser automation (Preview)
            **kwargs: Additional arguments passed to GenerateContentConfig constructor.
        """
        super().__init__()
        self.model = model
        self.events.register_events_from_module(events)

        if client is not None:
            self.client = client
        else:
            self.client = Client(api_key=api_key).aio

        # Video processing configuration
        self._fps = fps
        self._frame_format = frame_format
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._video_forwarder: Optional[VideoForwarder] = None
        self._enable_vision_tools = enable_vision_tools
        self._enable_structured_annotations = enable_structured_annotations

        # Buffer latest N seconds of video (only used if fps > 0)
        buffer_size = max(1, int(fps * frame_buffer_seconds)) if fps > 0 else 1
        self._frame_buffer: deque[av.VideoFrame] = deque(maxlen=buffer_size)

        # Store latest frame for on-demand access (used when fps=0)
        self._latest_frame: Optional[av.VideoFrame] = None
        self._latest_frame_lock = asyncio.Lock()

        # Thread pool for frame conversion
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # Register vision tools if enabled
        if self._enable_vision_tools:
            self._register_vision_tools()

        # Gemini-specific configuration
        self.thinking_level = thinking_level
        self.media_resolution = media_resolution
        self._builtin_tools = tools or []

        if config is not None:
            self._base_config: Optional[GenerateContentConfig] = config
        elif kwargs:
            self._base_config = GenerateContentConfig(**kwargs)
        else:
            self._base_config = None

        self.chat: Optional[Any] = None

    def _build_config(
        self,
        system_instruction: Optional[str] = None,
        base_config: Optional[GenerateContentConfig] = None,
    ) -> GenerateContentConfig:
        """
        Build GenerateContentConfig with Gemini features and built-in tools.

        Args:
            system_instruction: Optional system instruction to include.
            base_config: Optional base config to extend (takes precedence over self._base_config)

        Returns:
            GenerateContentConfig with thinking_level, media_resolution, and tools if set
        """
        if base_config is not None:
            config = base_config
        elif self._base_config is not None:
            config = self._base_config
        else:
            config = GenerateContentConfig()

        # Always include system instruction
        effective_instruction = (
            system_instruction if system_instruction else self._instructions
        )

        # Append annotation format instructions if enabled
        if self._enable_structured_annotations and effective_instruction:
            annotation_instructions = (
                "\n\nIMPORTANT: When asked to mark, circle, highlight, box, or annotate objects in the video, "
                "respond with a JSON array using your native bounding box format:\n"
                "```json\n"
                '[\n  {"box_2d": [y_min, x_min, y_max, x_max], "label": "object_name"}\n'
                "]\n```\n"
                "Use your standard coordinate format (normalized 0-1000, y-coordinate first). "
                "After the JSON, add a short friendly confirmation like 'I've marked that for you!'"
            )
            effective_instruction += annotation_instructions

        if effective_instruction:
            config.system_instruction = effective_instruction

        if self.thinking_level:
            from google.genai.types import ThinkingConfig

            config.thinking_config = ThinkingConfig(thinking_level=self.thinking_level)

        if self.media_resolution:
            config.media_resolution = self.media_resolution

        # Add built-in tools if configured
        if self._builtin_tools:
            builtin_tool_objects: list[types.Tool] = [
                tool.to_tool() for tool in self._builtin_tools
            ]
            if config.tools is None:
                config.tools = builtin_tool_objects  # type: ignore[assignment]
            else:
                # Append to existing tools
                existing_tools = list(config.tools)
                existing_tools.extend(builtin_tool_objects)
                config.tools = existing_tools  # type: ignore[assignment]

        return config

    def _register_vision_tools(self) -> None:
        """
        Register built-in vision tools that the LLM can call to capture frames.

        This is automatically called when enable_vision_tools=True.
        Registers two functions:
        - capture_frame(): Captures and returns the current video frame
        - analyze_video(question): Analyzes current frame with a specific question
        """

        @self.register_function(
            description=(
                "Capture the current frame from the video feed. "
                "Use this when the user asks about what they're showing, "
                "what you see, or to analyze their video. "
                "Returns a description that the frame was captured successfully."
            )
        )
        async def capture_frame() -> Dict[str, Any]:
            """
            Capture the current video frame for analysis.

            This function is automatically called by the LLM when it needs
            to see the video. The captured frame is included in the next
            message to Gemini.

            Returns:
                dict: Status message indicating frame was captured
            """
            if self._latest_frame is None:
                return {
                    "status": "error",
                    "message": "No video frame available. Camera may be off.",
                }

            return {
                "status": "success",
                "message": "Video frame captured. You can now analyze it.",
                "note": "The frame has been captured and will be included in your next response.",
            }

        @self.register_function(
            description=(
                "Analyze the current video frame to answer a specific question. "
                "Use this when you need to look at the video to answer the user. "
                "For example: 'What color is my shirt?', 'How many fingers am I holding up?'"
            )
        )
        async def analyze_video(question: str) -> Dict[str, Any]:
            """
            Capture and analyze the current video frame with a specific question.

            Args:
                question: The specific question to answer about the video frame

            Returns:
                dict: Status message with the question to be analyzed
            """
            if self._latest_frame is None:
                return {
                    "status": "error",
                    "message": "No video frame available. Camera may be off.",
                }

            return {
                "status": "success",
                "message": "Video frame captured for analysis",
                "question": question,
                "note": "The frame will be analyzed in your next response to answer: "
                + question,
            }

        logger.info(
            f"✅ Registered vision tools: capture_frame(), analyze_video() for {PLUGIN_NAME}"
        )

    async def _get_frame_for_tool_call(self) -> Optional[av.VideoFrame]:
        """
        Get the latest frame for on-demand tool calls.

        Returns:
            The latest video frame if available, None otherwise
        """
        async with self._latest_frame_lock:
            return self._latest_frame

    async def _convert_frame_to_part(
        self, frame: av.VideoFrame
    ) -> Optional[types.Part]:
        """
        Convert a video frame to a Gemini Part object.

        Args:
            frame: The video frame to convert

        Returns:
            Part object with the frame data, or None if conversion fails
        """
        try:
            loop = asyncio.get_running_loop()
            conversion_func = (
                frame_to_jpeg_bytes
                if self._frame_format == "jpeg"
                else frame_to_png_bytes
            )

            # Convert frame in thread pool
            frame_bytes = await loop.run_in_executor(
                self._executor,
                conversion_func,
                frame,
                self._frame_width,
                self._frame_height,
                85 if self._frame_format == "jpeg" else None,
            )

            mime_type = f"image/{self._frame_format}"
            return types.Part.from_bytes(data=frame_bytes, mime_type=mime_type)
        except Exception as e:
            logger.error(f"Failed to convert frame to Part: {e}")
            return None

    async def simple_response(
        self,
        text: str,
        processors: Optional[List[Processor]] = None,
        participant: Optional[Participant] = None,
    ) -> LLMResponseEvent[Any]:
        """
        Simple response is a standardized way to create an LLM response with video context.

        This method is called every time a new STT transcript is received. It automatically
        includes buffered video frames in the request to Gemini.

        Args:
            text: The text to respond to.
            processors: List of processors (which contain state) about the video/voice AI.
            participant: The Participant object, optional.

        Examples:

            llm.simple_response("What do you see in the video?")
        """
        if self._conversation is None:
            logger.warning(
                f'Cannot request a response from the LLM "{self.model}" - the conversation has not been initialized yet.'
            )
            return LLMResponseEvent(original=None, text="")

        # Add user message to conversation
        if participant is None:
            await self._conversation.send_message(
                role="user", user_id="user", content=text
            )

        # Count frames being processed and capture reference frame for annotations
        frames_count = len(self._frame_buffer)
        inference_id = str(uuid.uuid4())

        # Capture reference frame (most recent) for annotation drawing
        # This ensures annotations are drawn on the exact frame Gemini analyzed
        reference_frame: Optional[av.VideoFrame] = None
        if self._frame_buffer:
            reference_frame = self._frame_buffer[-1]  # Most recent frame
        elif self._latest_frame:
            reference_frame = self._latest_frame

        # Emit VLM start event
        self.events.send(
            VLMInferenceStartEvent(
                plugin_name=PLUGIN_NAME,
                inference_id=inference_id,
                model=self.model,
                frames_count=frames_count,
            )
        )

        # Emit request started event
        self.events.send(
            LLMRequestStartedEvent(
                plugin_name=PLUGIN_NAME,
                model=self.model,
                streaming=True,
            )
        )

        # Track timing
        request_start_time = time.perf_counter()
        first_token_time: Optional[float] = None

        try:
            # Check if we should include frame for on-demand mode
            # In on-demand mode (fps=0), we only send frames after a vision tool was called
            # The tool call response will trigger a follow-up that includes the frame
            include_frame = self._fps > 0  # Automatic mode always includes frames

            # Build message with text and video frames
            message_parts = await self._build_message_with_frames(
                text, include_latest_frame=include_frame
            )

            # Initialize chat if needed
            if self.chat is None:
                config = self._build_config(system_instruction=self._instructions)
                self.chat = self.client.chats.create(model=self.model, config=config)

            # Add tools if available
            config = None
            tools_spec = self.get_available_functions()
            if tools_spec:
                conv_tools = self._convert_tools_to_provider_format(tools_spec)
                config = self._build_config()
                config.tools = conv_tools  # type: ignore[assignment]
            elif self.thinking_level or self.media_resolution:
                config = self._build_config()

            # Send message to Gemini
            iterator = await self.chat.send_message_stream(message_parts, config=config)
            text_parts: List[str] = []
            final_chunk = None
            pending_calls: List[NormalizedToolCallItem] = []

            # Gemini API does not have an item_id, we create it here
            item_id = str(uuid.uuid4())

            idx = 0
            async for chunk in iterator:
                final_chunk = chunk

                # Track time to first token
                if first_token_time is None and hasattr(chunk, "text") and chunk.text:
                    first_token_time = time.perf_counter()

                self._standardize_and_emit_event(
                    chunk,
                    text_parts,
                    item_id,
                    idx,
                    request_start_time=request_start_time,
                    first_token_time=first_token_time,
                )

                # Collect function calls as they stream
                try:
                    chunk_calls = self._extract_tool_calls_from_stream_chunk(chunk)
                    pending_calls.extend(chunk_calls)
                except Exception:
                    pass

                idx += 1

            # Handle function calls if present
            if pending_calls:
                # Multi-hop tool calling loop
                MAX_ROUNDS = 3
                rounds = 0
                current_calls = pending_calls
                cfg_with_tools = config

                seen: set[str] = set()
                vision_tool_called = False

                while current_calls and rounds < MAX_ROUNDS:
                    # Execute tools concurrently with deduplication
                    triples, seen = await self._dedup_and_execute(
                        current_calls, max_concurrency=8, timeout_s=30, seen=seen
                    )  # type: ignore[arg-type]

                    parts = []
                    for tc, res, err in triples:
                        # Check if a vision tool was called
                        if tc["name"] in ("capture_frame", "analyze_video"):
                            vision_tool_called = True

                        # Ensure response is a dictionary
                        if not isinstance(res, dict):
                            res = {"result": res}

                        # Sanitize large outputs
                        sanitized_res = {}
                        for k, v in res.items():
                            sanitized_res[k] = self._sanitize_tool_output(v)

                        # Create function response part
                        func_response_part = types.Part.from_function_response(
                            name=tc["name"], response=sanitized_res
                        )

                        # Include thought signature for Gemini 3 Pro compatibility
                        if (
                            "thought_signature" in tc
                            and tc["thought_signature"] is not None
                        ):
                            func_response_part.thought_signature = tc[
                                "thought_signature"
                            ]

                        parts.append(func_response_part)

                    # If vision tool was called in on-demand mode, include the latest frame
                    if vision_tool_called and self._fps == 0 and self._latest_frame:
                        logger.debug(
                            "Vision tool called - including latest frame in follow-up"
                        )
                        frame_part = await self._convert_frame_to_part(
                            self._latest_frame
                        )
                        if frame_part:
                            parts.append(frame_part)
                        vision_tool_called = False  # Reset for next round

                    # Fix for Gemini 3 Pro: Remove empty model messages from history
                    if self._is_gemini_3_model():
                        await self._clean_chat_history_for_gemini_3()

                    # Send function responses with tools config
                    follow_up_iter = await self.chat.send_message_stream(
                        parts, config=cfg_with_tools
                    )
                    follow_up_text_parts: List[str] = []
                    follow_up_last = None
                    next_calls = []
                    follow_up_idx = 0

                    async for chk in follow_up_iter:
                        follow_up_last = chk
                        self._standardize_and_emit_event(
                            chk,
                            follow_up_text_parts,
                            item_id,
                            follow_up_idx,
                            request_start_time=request_start_time,
                            first_token_time=first_token_time,
                        )

                        # Check for new function calls
                        try:
                            chunk_calls = self._extract_tool_calls_from_stream_chunk(
                                chk
                            )
                            next_calls.extend(chunk_calls)
                        except Exception:
                            pass

                        follow_up_idx += 1

                    current_calls = next_calls
                    rounds += 1

                total_text = "".join(follow_up_text_parts) or "".join(text_parts)
                llm_response = LLMResponseEvent(follow_up_last or final_chunk, total_text)
            else:
                total_text = "".join(text_parts)
                llm_response = LLMResponseEvent(final_chunk, total_text)

            # Calculate timing metrics
            latency_ms = (time.perf_counter() - request_start_time) * 1000
            ttft_ms: Optional[float] = None
            if first_token_time is not None:
                ttft_ms = (first_token_time - request_start_time) * 1000

            # Extract token usage from response if available
            input_tokens: Optional[int] = None
            output_tokens: Optional[int] = None
            if (
                final_chunk
                and hasattr(final_chunk, "usage_metadata")
                and final_chunk.usage_metadata
            ):
                usage = final_chunk.usage_metadata
                input_tokens = getattr(usage, "prompt_token_count", None)
                output_tokens = getattr(usage, "candidates_token_count", None)

            # Emit VLM-specific completion event
            self.events.send(
                VLMInferenceCompletedEvent(
                    plugin_name=PLUGIN_NAME,
                    inference_id=inference_id,
                    model=self.model,
                    text=total_text,
                    latency_ms=latency_ms,
                    frames_processed=frames_count,
                )
            )

            # Emit LLM completion event
            self.events.send(
                LLMResponseCompletedEvent(
                    plugin_name=PLUGIN_NAME,
                    original=llm_response.original,
                    text=llm_response.text,
                    item_id=item_id,
                    latency_ms=latency_ms,
                    time_to_first_token_ms=ttft_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=(input_tokens or 0) + (output_tokens or 0)
                    if input_tokens or output_tokens
                    else None,
                    model=self.model,
                )
            )

            # Check for annotations in response and emit VLMAnnotationEvent
            # This allows AnnotationProcessor to draw on the correct reference frame
            if reference_frame is not None and self._contains_annotations(total_text):
                annotation_json = self._extract_annotation_json(total_text)
                if annotation_json:
                    logger.debug(
                        f"📐 Emitting VLMAnnotationEvent with reference frame"
                    )
                    self.events.send(
                        VLMAnnotationEvent(
                            plugin_name=PLUGIN_NAME,
                            annotations_json=annotation_json,
                            reference_frame=reference_frame,
                            inference_id=inference_id,
                        )
                    )

            return llm_response

        except Exception as e:
            logger.exception(f'Failed to get a response from the model "{self.model}"')
            self.events.send(
                events.LLMErrorEvent(
                    plugin_name=PLUGIN_NAME,
                    error_message=str(e),
                    event_data=e,
                )
            )
            self.events.send(
                VLMErrorEvent(
                    plugin_name=PLUGIN_NAME,
                    inference_id=inference_id,
                    error=e,
                    context="api_request",
                )
            )
            return LLMResponseEvent(original=None, text="", exception=e)

    async def _build_message_with_frames(
        self, text: str, include_latest_frame: bool = False
    ) -> List[types.Part]:
        """
        Build a message with text and video frames.

        Behavior depends on mode:
        - fps > 0 (automatic buffering): Includes all buffered frames
        - fps = 0 (on-demand): Only includes frame if include_latest_frame=True

        Args:
            text: The text content of the message.
            include_latest_frame: If True and fps=0, includes the latest frame.
                Used when vision tools are called.

        Returns:
            List of Part objects containing text and video frames.
        """
        parts: List[types.Part] = []

        # Add text part
        parts.append(types.Part.from_text(text=text))

        # Mode 1: Automatic buffering (fps > 0)
        if self._fps > 0 and self._frame_buffer:
            logger.debug(
                f'Forwarding {len(self._frame_buffer)} buffered frames to "{self.model}"'
            )

            # Convert frames to bytes in parallel
            loop = asyncio.get_running_loop()
            conversion_func = (
                frame_to_jpeg_bytes
                if self._frame_format == "jpeg"
                else frame_to_png_bytes
            )

            coroutines = [
                loop.run_in_executor(
                    self._executor,
                    conversion_func,
                    frame,
                    self._frame_width,
                    self._frame_height,
                    85 if self._frame_format == "jpeg" else None,
                )
                for frame in self._frame_buffer
            ]

            frame_bytes_list = await asyncio.gather(*coroutines)

            # Create Part objects for each frame
            mime_type = f"image/{self._frame_format}"
            for frame_bytes in frame_bytes_list:
                parts.append(
                    types.Part.from_bytes(data=frame_bytes, mime_type=mime_type)
                )

        # Mode 2: On-demand capture (fps = 0)
        elif self._fps == 0 and include_latest_frame and self._latest_frame:
            logger.debug(
                f'Forwarding latest frame on-demand to "{self.model}" (triggered by tool call)'
            )
            frame_part = await self._convert_frame_to_part(self._latest_frame)
            if frame_part:
                parts.append(frame_part)

        return parts

    def _contains_annotations(self, text: str) -> bool:
        """
        Check if the text contains annotation JSON (box_2d, circle, polygon, etc.).

        Args:
            text: The response text to check.

        Returns:
            True if annotations are likely present.
        """
        # Look for annotation keywords in JSON-like context
        annotation_keywords = ["box_2d", '"circle"', '"polygon"', '"point"', '"line"']
        return any(kw in text for kw in annotation_keywords)

    def _extract_annotation_json(self, text: str) -> str:
        """
        Extract annotation JSON from response text.

        Looks for JSON arrays containing annotation data, either in markdown
        code blocks or inline.

        Args:
            text: The response text containing annotations.

        Returns:
            The extracted JSON string, or empty string if none found.
        """
        # Try markdown code blocks first
        markdown_pattern = r"```json\s*([\s\S]*?)\s*```"
        markdown_matches = re.findall(markdown_pattern, text)

        for match in markdown_matches:
            try:
                data = json.loads(match)
                if isinstance(data, list) and any(
                    "box_2d" in item or "circle" in item or "polygon" in item
                    for item in data
                    if isinstance(item, dict)
                ):
                    return match
            except json.JSONDecodeError:
                continue

        # Try inline JSON arrays
        inline_pattern = r"\[\s*\{[^\]]+\]\s*"
        inline_matches = re.findall(inline_pattern, text, re.DOTALL)

        for match in inline_matches:
            try:
                data = json.loads(match)
                if isinstance(data, list) and any(
                    "box_2d" in item or "circle" in item or "polygon" in item
                    for item in data
                    if isinstance(item, dict)
                ):
                    return match
            except json.JSONDecodeError:
                continue

        return ""

    async def _on_frame_received(self, frame: av.VideoFrame) -> None:
        """
        Callback to handle received video frames.

        In automatic mode (fps > 0), adds frames to buffer.
        In on-demand mode (fps = 0), stores only the latest frame.
        """
        if self._fps > 0:
            # Automatic buffering mode
            self._frame_buffer.append(frame)
        else:
            # On-demand mode - store latest frame for tool calls
            async with self._latest_frame_lock:
                self._latest_frame = frame

    async def watch_video_track(
        self,
        track: MediaStreamTrack,
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        """
        Setup video forwarding and start buffering video frames.
        This method is called by the `Agent`.

        Behavior:
        - fps > 0: Continuously buffers frames at specified FPS
        - fps = 0: Only stores latest frame for on-demand capture

        Args:
            track: Instance of VideoStreamTrack.
            shared_forwarder: A shared VideoForwarder instance if present. Defaults to None.

        Returns: None
        """
        if self._video_forwarder is not None and shared_forwarder is None:
            logger.warning("Video forwarder already running, stopping the previous one")
            await self._video_forwarder.stop()
            self._video_forwarder = None
            logger.info("Stopped video forwarding")

        mode = "automatic buffering" if self._fps > 0 else "on-demand capture"
        logger.info(
            f'🎥 Subscribing plugin "{PLUGIN_NAME}" to VideoForwarder (mode: {mode}, fps: {self._fps})'
        )

        if shared_forwarder:
            self._video_forwarder = shared_forwarder
        else:
            # For on-demand mode (fps=0), use minimal FPS for VideoForwarder
            # but capture at low rate just to keep latest frame
            forwarder_fps = max(0.5, self._fps) if self._fps == 0 else self._fps
            self._video_forwarder = VideoForwarder(
                cast(VideoStreamTrack, track),
                max_buffer=10,
                fps=forwarder_fps,
                name=f"{PLUGIN_NAME}_forwarder",
            )
            self._video_forwarder.start()

        # Add frame handler with custom callback
        self._video_forwarder.add_frame_handler(
            self._on_frame_received,
            fps=max(0.5, self._fps) if self._fps == 0 else self._fps,
        )

    async def stop_watching_video_track(self) -> None:
        """Stop watching the video track."""
        if self._video_forwarder is not None:
            await self._video_forwarder.remove_frame_handler(self._on_frame_received)
            self._video_forwarder = None
            logger.info(
                f"🛑 Stopped video forwarding to {PLUGIN_NAME} (participant left)"
            )

    def _standardize_and_emit_event(
        self,
        chunk: Any,
        text_parts: List[str],
        item_id: str,
        idx: int,
        request_start_time: Optional[float] = None,
        first_token_time: Optional[float] = None,
    ) -> None:
        """
        Forward native events and send standardized versions.
        """
        # Forward the native event
        self.events.send(
            events.GeminiResponseEvent(plugin_name=PLUGIN_NAME, response_chunk=chunk)
        )

        # Extract text directly from parts
        chunk_text = self._extract_text_from_chunk(chunk)
        if chunk_text:
            self.events.send(
                LLMResponseChunkEvent(
                    plugin_name=PLUGIN_NAME,
                    content_index=idx,
                    item_id=item_id,
                    delta=chunk_text,
                )
            )
            text_parts.append(chunk_text)

    @staticmethod
    def _extract_text_from_chunk(chunk: Any) -> str:
        """Extract text from response chunk without triggering SDK warning."""
        texts = []
        if hasattr(chunk, "candidates") and chunk.candidates:
            for candidate in chunk.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.text:
                            texts.append(part.text)
        return "".join(texts)

    def _convert_tools_to_provider_format(
        self, tools: List[ToolSchema]
    ) -> List[Dict[str, Any]]:
        """
        Convert ToolSchema objects to Gemini format.

        Args:
            tools: List of ToolSchema objects

        Returns:
            List of tools in Gemini format
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

        return [{"function_declarations": function_declarations}]

    def _extract_tool_calls_from_stream_chunk(
        self, chunk: Any
    ) -> List[NormalizedToolCallItem]:
        """
        Extract tool calls from Gemini streaming chunk.

        Args:
            chunk: Gemini streaming event

        Returns:
            List of normalized tool call items
        """
        calls: List[NormalizedToolCallItem] = []

        try:
            if hasattr(chunk, "candidates") and chunk.candidates:
                for c in chunk.candidates:
                    if c.content:
                        for part in c.content.parts:
                            if part.function_call:
                                thought_sig = part.thought_signature
                                call_item: NormalizedToolCallItem = {
                                    "type": "tool_call",
                                    "name": part.function_call.name,
                                    "arguments_json": part.function_call.args,
                                }
                                if thought_sig is not None:
                                    call_item["thought_signature"] = thought_sig
                                calls.append(call_item)
        except Exception:
            pass

        return calls

    def _is_gemini_3_model(self) -> bool:
        """Check if the current model is Gemini 3."""
        return "gemini-3" in self.model.lower()

    async def _clean_chat_history_for_gemini_3(self) -> None:
        """
        Clean chat history for Gemini 3 Pro by removing empty model messages.

        Gemini 3 Pro streaming returns an extra empty content chunk after function calls,
        which the SDK records as an empty model message in history. This breaks the
        requirement that "function response turn comes immediately after function call turn".
        """
        if not self.chat:
            return

        # Get current history
        history = self.chat.get_history()

        # Filter out empty model messages
        cleaned_history = []
        for content in history:
            if content.role == "model":
                if content.parts:
                    has_meaningful_content = False
                    for part in content.parts:
                        if (
                            part.function_call
                            or part.function_response
                            or (part.text and len(part.text) > 0)
                        ):
                            has_meaningful_content = True
                            break

                    if has_meaningful_content:
                        cleaned_history.append(content)
            else:
                cleaned_history.append(content)

        # If we filtered anything out, recreate the chat with cleaned history
        if len(cleaned_history) < len(history):
            config = self._build_config(system_instruction=self._instructions)
            self.chat = self.client.chats.create(
                model=self.model, config=config, history=cleaned_history
            )

    async def close(self) -> None:
        """Clean up resources."""
        await self.stop_watching_video_track()
        self._executor.shutdown(wait=False)
