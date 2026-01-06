import uuid
from typing import Optional, List, TYPE_CHECKING, Any, Dict, AsyncIterator

from google.genai.client import AsyncClient, Client
from google.genai import types
from google.genai.types import (
    GenerateContentResponse,
    GenerateContentConfig,
    ThinkingLevel,
    MediaResolution,
)

from vision_agents.core.llm.llm import LLM, LLMResponseEvent
from vision_agents.core.llm.llm_types import ToolSchema, NormalizedToolCallItem

from vision_agents.core.llm.events import (
    LLMResponseCompletedEvent,
    LLMResponseChunkEvent,
)

from . import events
from .tools import GeminiTool

from vision_agents.core.processors import Processor

if TYPE_CHECKING:
    from vision_agents.core.agents.conversation import Message


DEFAULT_MODEL = "gemini-3-pro-preview"


class GeminiLLM(LLM):
    """
    The GeminiLLM class provides full/native access to the gemini SDK methods.
    It only standardized the minimal feature set that's needed for the agent integration.

    The agent requires that we standardize:
    - sharing instructions
    - keeping conversation history
    - response normalization

    Notes on the Gemini integration:
    - the native method is called send_message (maps 1-1 to chat.send_message_stream)
    - history is maintained in the gemini sdk (with the usage of client.chats.create(model=self.model))

    Examples:

          from vision_agents.plugins import gemini
          llm = gemini.LLM()
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        client: Optional[AsyncClient] = None,
        thinking_level: Optional[ThinkingLevel] = None,
        media_resolution: Optional[MediaResolution] = None,
        config: Optional[GenerateContentConfig] = None,
        tools: Optional[List[GeminiTool]] = None,
        **kwargs,
    ):
        """
        Initialize the GeminiLLM class.

        Args:
            model (str): The model to use. Defaults to models/gemini-3-pro-preview.
            api_key: optional API key. by default loads from GOOGLE_API_KEY
            client: optional Gemini client. by default creates a new client object.
            thinking_level: Optional thinking level for Gemini 3. Use ThinkingLevel.LOW or
                ThinkingLevel.HIGH. Defaults to "high" for Gemini 3 Pro if not specified.
                Cannot be used with legacy thinking_budget parameter.
            media_resolution: Optional media resolution for multimodal processing. Use
                MediaResolution.MEDIA_RESOLUTION_LOW, MEDIA_RESOLUTION_MEDIUM, or
                MediaResolution.MEDIA_RESOLUTION_HIGH. Recommended: "high" for images, "medium" for PDFs,
                "low"/"medium" for general video, "high" for text-heavy video.
            config: Optional[GenerateContentConfig] to use as base. Any kwargs will be passed
                to GenerateContentConfig constructor if config is not provided.
            tools: Optional list of Gemini built-in tools. Available tools:
                - tools.FileSearch(store): RAG over your documents
                - tools.GoogleSearch(): Ground responses with web data
                - tools.CodeExecution(): Run Python code
                - tools.URLContext(): Read specific web pages
                - tools.GoogleMaps(): Location-aware queries (Preview)
                - tools.ComputerUse(): Browser automation (Preview)
                See: https://ai.google.dev/gemini-api/docs/tools
            **kwargs: Additional arguments passed to GenerateContentConfig constructor.
        """
        super().__init__()
        self.events.register_events_from_module(events)
        self.model = model
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

        if client is not None:
            self.client = client
        else:
            self.client = Client(api_key=api_key).aio

    def _build_config(
        self,
        system_instruction: Optional[str] = None,
        base_config: Optional[GenerateContentConfig] = None,
    ) -> GenerateContentConfig:
        """
        Build GenerateContentConfig with Gemini 3 features and built-in tools.

        Args:
            system_instruction: Optional system instruction to include. If not provided,
                uses self._instructions to ensure instructions are always passed.
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

        # Always include system instruction - passing any config to send_message_stream
        # overrides the chat-level system instruction, so we must include it every time
        effective_instruction = (
            system_instruction if system_instruction else self._instructions
        )
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

    async def simple_response(
        self,
        text: str,
        processors: Optional[List[Processor]] = None,
        participant: Optional[Any] = None,
    ) -> LLMResponseEvent[Any]:
        """
        simple_response is a standardized way (across openai, claude, gemini etc.) to create a response.

        Args:
            text: The text to respond to
            processors: list of processors (which contain state) about the video/voice AI

        Examples:

            llm.simple_response("say hi to the user, be mean")
        """
        return await self.send_message(message=text)

    async def send_message(self, *args, **kwargs):
        """
        send_message gives you full support/access to the native Gemini chat send message method
        under the hood it calls chat.send_message_stream(*args, **kwargs)
        this method wraps and ensures we broadcast an event which the agent class hooks into
        """
        # if "model" not in kwargs:
        #    kwargs["model"] = self.model

        # initialize chat if needed
        if self.chat is None:
            config = self._build_config(system_instruction=self._instructions)
            self.chat = self.client.chats.create(model=self.model, config=config)

        # Add tools if available - Gemini uses GenerateContentConfig
        tools_spec = self.get_available_functions()
        if tools_spec:
            conv_tools = self._convert_tools_to_provider_format(tools_spec)
            cfg = kwargs.get("config")
            if not isinstance(cfg, GenerateContentConfig):
                cfg = self._build_config()
            else:
                cfg = self._build_config(base_config=cfg)
            cfg.tools = conv_tools  # type: ignore[assignment]
            kwargs["config"] = cfg
        elif self.thinking_level or self.media_resolution:
            # Only pass config if we need to set thinking_level or media_resolution
            # Don't pass an empty config as it overrides the system_instruction from chat creation
            cfg = kwargs.get("config")
            if cfg is None or not isinstance(cfg, GenerateContentConfig):
                cfg = self._build_config()
            else:
                cfg = self._build_config(base_config=cfg)
            kwargs["config"] = cfg
        # If no tools and no thinking/media config needed, don't pass config
        # This preserves the system_instruction set during chat creation

        # Generate content using the client
        iterator: AsyncIterator[
            GenerateContentResponse
        ] = await self.chat.send_message_stream(*args, **kwargs)
        text_parts: List[str] = []
        final_chunk = None
        pending_calls: List[NormalizedToolCallItem] = []

        # Gemini API does not have an item_id, we create it here and add it to all events
        item_id = str(uuid.uuid4())

        idx = 0
        async for chunk in iterator:
            response_chunk: GenerateContentResponse = chunk
            final_chunk = response_chunk
            self._standardize_and_emit_event(response_chunk, text_parts, item_id, idx)

            # collect function calls as they stream
            try:
                chunk_calls = self._extract_tool_calls_from_stream_chunk(chunk)
                pending_calls.extend(chunk_calls)
            except Exception:
                pass  # Ignore errors in chunk processing

            idx += 1

        # Check if there were function calls in the response
        if pending_calls:
            # Multi-hop tool calling loop
            MAX_ROUNDS = 3
            rounds = 0
            current_calls = pending_calls
            cfg_with_tools = kwargs.get("config")

            seen: set[str] = set()
            while current_calls and rounds < MAX_ROUNDS:
                # Execute tools concurrently with deduplication
                triples, seen = await self._dedup_and_execute(
                    current_calls, max_concurrency=8, timeout_s=30, seen=seen
                )  # type: ignore[arg-type]

                executed = []
                parts = []
                for tc, res, err in triples:
                    executed.append(tc)
                    # Ensure response is a dictionary for Gemini and sanitize output
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
                    # The thought signature from the function call must be included in the response
                    if (
                        "thought_signature" in tc
                        and tc["thought_signature"] is not None
                    ):
                        func_response_part.thought_signature = tc["thought_signature"]

                    parts.append(func_response_part)

                # Fix for Gemini 3 Pro: Remove empty model messages from history
                # Gemini 3 Pro streaming adds an empty model message after function calls
                # which breaks the "function response must immediately follow function call" requirement
                if self._is_gemini_3_model():
                    await self._clean_chat_history_for_gemini_3()

                # Send function responses with tools config
                follow_up_iter: AsyncIterator[
                    GenerateContentResponse
                ] = await self.chat.send_message_stream(parts, config=cfg_with_tools)  # type: ignore[arg-type]
                follow_up_text_parts: List[str] = []
                follow_up_last = None
                next_calls = []
                follow_up_idx = 0

                async for chk in follow_up_iter:
                    follow_up_last = chk
                    # TODO: unclear if this is correct (item_id and idx)
                    self._standardize_and_emit_event(
                        chk, follow_up_text_parts, item_id, follow_up_idx
                    )

                    # Check for new function calls
                    try:
                        chunk_calls = self._extract_tool_calls_from_stream_chunk(chk)
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

        self.events.send(
            LLMResponseCompletedEvent(
                plugin_name="gemini",
                original=llm_response.original,
                text=llm_response.text,
                item_id=item_id,
            )
        )

        # Return the LLM response
        return llm_response

    @staticmethod
    def _normalize_message(gemini_input) -> List["Message"]:
        from vision_agents.core.agents.conversation import Message

        # standardize on input
        if isinstance(gemini_input, str):
            gemini_input = [gemini_input]

        if not isinstance(gemini_input, List):
            gemini_input = [gemini_input]

        messages = []
        for i in gemini_input:
            message = Message(original=i, content=i)
            messages.append(message)

        return messages

    def _standardize_and_emit_event(
        self,
        chunk: GenerateContentResponse,
        text_parts: List[str],
        item_id: str,
        idx: int,
    ) -> Optional[LLMResponseEvent[Any]]:
        """
        Forwards the events and also send out a standardized version (the agent class hooks into that)
        """
        # forward the native event
        self.events.send(
            events.GeminiResponseEvent(plugin_name="gemini", response_chunk=chunk)
        )

        # Check if response has text content
        if hasattr(chunk, "text") and chunk.text:
            self.events.send(
                LLMResponseChunkEvent(
                    plugin_name="gemini",
                    content_index=idx,
                    item_id=item_id,
                    delta=chunk.text,
                )
            )
            text_parts.append(chunk.text)

        return None

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

        # Return as dict with function_declarations (SDK accepts dicts)
        return [{"function_declarations": function_declarations}]

    def _extract_tool_calls_from_response(
        self, response: Any
    ) -> List[NormalizedToolCallItem]:
        """
        Extract tool calls from Gemini response.

        Args:
            response: Gemini response object

        Returns:
            List of normalized tool call items with thought signatures for Gemini 3
        """
        calls: List[NormalizedToolCallItem] = []

        try:
            # We must iterate through candidates to get the thought_signature
            # The top-level response.function_calls convenience property returns FunctionCall objects
            # which do not have the thought_signature attribute.
            if response.candidates:
                for c in response.candidates:
                    if c.content:
                        for part in c.content.parts:
                            if part.function_call:
                                # Extract thought signature for Gemini 3 Pro compatibility
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
            pass  # Ignore extraction errors

        return calls

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
        try:
            return self._extract_tool_calls_from_response(
                chunk
            )  # chunks use same shape
        except Exception:
            return []  # Ignore extraction errors

    def _create_tool_result_parts(
        self, tool_calls: List[NormalizedToolCallItem], results: List[Any]
    ):
        """
        Create function_response parts for Gemini.

        Args:
            tool_calls: List of tool calls that were executed
            results: List of results from function execution

        Returns:
            List of function_response parts
        """
        parts = []
        for tc, res in zip(tool_calls, results):
            try:
                # Convert result to dict if it's not already
                if isinstance(res, dict):
                    response_data = res
                else:
                    response_data = {"result": res}

                # res may be dict/list/str; pass directly; SDK serializes
                parts.append(
                    types.Part.from_function_response(
                        name=tc["name"], response=response_data
                    )
                )
            except Exception:
                # Fallback: create a simple text part
                parts.append(types.Part(text=f"Function {tc['name']} returned: {res}"))
        return parts

    def _is_gemini_3_model(self) -> bool:
        """Check if the current model is Gemini 3."""
        return "gemini-3" in self.model.lower()

    async def _clean_chat_history_for_gemini_3(self) -> None:
        """
        Clean chat history for Gemini 3 Pro by removing empty model messages.

        Gemini 3 Pro streaming returns an extra empty content chunk after function calls,
        which the SDK records as an empty model message in history. This breaks the
        requirement that "function response turn comes immediately after function call turn".

        This method:
        1. Gets current chat history
        2. Filters out empty model messages
        3. Recreates the chat with cleaned history
        """
        if not self.chat:
            return

        # Get current history
        history = self.chat.get_history()

        # Filter out empty model messages
        # An empty message has no meaningful content (no text, no function_call, etc.)
        cleaned_history = []
        for content in history:
            if content.role == "model":
                # Only keep model messages that have parts with meaningful content
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

                    # Only add model messages with meaningful content
                    if has_meaningful_content:
                        cleaned_history.append(content)
                # Skip model messages with no parts (they are empty)
            else:
                # Keep all non-model messages (e.g., user messages)
                cleaned_history.append(content)

        # If we filtered anything out, recreate the chat with cleaned history
        if len(cleaned_history) < len(history):
            config = self._build_config(system_instruction=self._instructions)
            self.chat = self.client.chats.create(
                model=self.model, config=config, history=cleaned_history
            )
