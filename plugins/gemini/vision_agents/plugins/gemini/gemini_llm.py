from typing import Optional, List, TYPE_CHECKING, Any, Dict

from google import genai
from google.genai import types
from google.genai.types import GenerateContentResponse

from vision_agents.core.llm.llm import LLM, LLMResponseEvent
from vision_agents.core.llm.llm_types import ToolSchema, NormalizedToolCallItem

from vision_agents.core.llm.events import LLMResponseCompletedEvent, LLMResponseChunkEvent

from . import events

from vision_agents.core.processors import Processor

if TYPE_CHECKING:
    from vision_agents.core.agents.conversation import Message


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
    def __init__(self, model: str, api_key: Optional[str] = None, client: Optional[genai.Client] = None):
        """
        Initialize the GeminiLLM class.

        Args:
            model (str): The model to use.
            api_key: optional API key. by default loads from GOOGLE_API_KEY
            client: optional Anthropic client. by default creates a new client object.
        """
        super().__init__()
        self.events.register_events_from_module(events)
        self.model = model
        self.chat: Optional[Any] = None

        if client is not None:
            self.client = client
        else:
            self.client = genai.Client(api_key=api_key)

    async def simple_response(self, text: str, processors: Optional[List[Processor]] = None, participant: Optional[Any] = None) -> LLMResponseEvent[Any]:
        """
        simple_response is a standardized way (across openai, claude, gemini etc.) to create a response.

        Args:
            text: The text to respond to
            processors: list of processors (which contain state) about the video/voice AI

        Examples:

            llm.simple_response("say hi to the user, be mean")
        """
        return await self.send_message(
            message=text
        )

    async def send_message(self, *args, **kwargs):
        """
        send_message gives you full support/access to the native Gemini chat send message method
        under the hood it calls chat.send_message_stream(*args, **kwargs)
        this method wraps and ensures we broadcast an event which the agent class hooks into
        """
        #if "model" not in kwargs:
        #    kwargs["model"] = self.model

        # initialize chat if needed
        if self.chat is None:
            self.chat = self.client.chats.create(model=self.model)

        # Add tools if available - Gemini uses GenerateContentConfig
        tools_spec = self.get_available_functions()
        if tools_spec:
            from google.genai import types
            conv_tools = self._convert_tools_to_provider_format(tools_spec)
            cfg = kwargs.get("config")
            if not isinstance(cfg, types.GenerateContentConfig):
                cfg = types.GenerateContentConfig()
            cfg.tools = conv_tools  # type: ignore[assignment]
            kwargs["config"] = cfg

        # Generate content using the client
        iterator = self.chat.send_message_stream(*args, **kwargs)
        text_parts : List[str] = []
        final_chunk = None
        pending_calls: List[NormalizedToolCallItem] = []
        
        for chunk in iterator:
            response_chunk: GenerateContentResponse = chunk
            final_chunk = response_chunk
            llm_response_optional = self._standardize_and_emit_event(response_chunk, text_parts)
            # collect function calls as they stream
            try:
                chunk_calls = self._extract_tool_calls_from_stream_chunk(chunk)
                pending_calls.extend(chunk_calls)
            except Exception:
                pass  # Ignore errors in chunk processing
            if llm_response_optional is not None:
                llm_response = llm_response_optional

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
                triples, seen = await self._dedup_and_execute(current_calls, max_concurrency=8, timeout_s=30, seen=seen)  # type: ignore[arg-type]
                
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
                    parts.append(types.Part.from_function_response(name=tc["name"], response=sanitized_res))
                
                # Send function responses with tools config
                follow_up_iter = self.chat.send_message_stream(parts, config=cfg_with_tools)  # type: ignore[arg-type]
                
                follow_up_text_parts: List[str] = []
                follow_up_last = None
                next_calls = []
                
                for chk in follow_up_iter:
                    follow_up_last = chk
                    llm_response_optional = self._standardize_and_emit_event(chk, follow_up_text_parts)
                    if llm_response_optional is not None:
                        llm_response = llm_response_optional
                    
                    # Check for new function calls
                    try:
                        chunk_calls = self._extract_tool_calls_from_stream_chunk(chk)
                        next_calls.extend(chunk_calls)
                    except Exception:
                        pass
                
                current_calls = next_calls
                rounds += 1

            total_text = "".join(follow_up_text_parts) or "".join(text_parts)
            llm_response = LLMResponseEvent(follow_up_last or final_chunk, total_text)
        else:
            total_text = "".join(text_parts)
            llm_response = LLMResponseEvent(final_chunk, total_text)

        self.events.send(LLMResponseCompletedEvent(
            plugin_name="gemini",
            original=llm_response.original,
            text=llm_response.text
        ))

        # Return the LLM response
        return llm_response

    @staticmethod
    def _normalize_message(gemini_input) -> List["Message"]:
        from vision_agents.core.agents.conversation import Message
        
        # standardize on input
        if isinstance(gemini_input, str):
            gemini_input = [
                gemini_input
            ]

        if not isinstance(gemini_input, List):
            gemini_input = [gemini_input]

        messages = []
        for i in gemini_input:
            message = Message(original=i, content=i)
            messages.append(message)

        return messages

    def _standardize_and_emit_event(self, chunk: GenerateContentResponse, text_parts: List[str]) -> Optional[LLMResponseEvent[Any]]:
        """
        Forwards the events and also send out a standardized version (the agent class hooks into that)
        """
        # forward the native event
        self.events.send(events.GeminiResponseEvent(
            plugin_name="gemini",
            response_chunk=chunk
        ))

        # Check if response has text content
        if hasattr(chunk, 'text') and chunk.text:
            self.events.send(LLMResponseChunkEvent(
                plugin_name="gemini",
                content_index=0,
                item_id="",
                output_index=0,
                sequence_number=0,
                delta=chunk.text,
            ))
            text_parts.append(chunk.text)

        return None

    def _convert_tools_to_provider_format(self, tools: List[ToolSchema]) -> List[Dict[str, Any]]:
        """
        Convert ToolSchema objects to Gemini format.
        Args:
            tools: List of ToolSchema objects
        Returns:
            List of tools in Gemini format
        """
        function_declarations = []
        for tool in tools:
            function_declarations.append({
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool["parameters_schema"]
            })
        
        # Return as dict with function_declarations (SDK accepts dicts)
        return [{"function_declarations": function_declarations}]

    def _extract_tool_calls_from_response(self, response: Any) -> List[NormalizedToolCallItem]:
        """
        Extract tool calls from Gemini response.
        
        Args:
            response: Gemini response object
            
        Returns:
            List of normalized tool call items
        """
        calls: List[NormalizedToolCallItem] = []
        
        try:
            # Prefer the top-level convenience list if available
            function_calls = getattr(response, "function_calls", []) or []
            for fc in function_calls:
                calls.append({
                    "type": "tool_call", 
                    "name": getattr(fc, "name", "unknown"), 
                    "arguments_json": getattr(fc, "args", {})
                })

            if not calls and getattr(response, "candidates", None):
                for c in response.candidates:
                    if getattr(c, "content", None):
                        for part in c.content.parts:
                            if getattr(part, "function_call", None):
                                calls.append({
                                    "type": "tool_call",
                                    "name": getattr(part.function_call, "name", "unknown"),
                                    "arguments_json": getattr(part.function_call, "args", {}),
                                })
        except Exception:
            pass  # Ignore extraction errors
        
        return calls

    def _extract_tool_calls_from_stream_chunk(self, chunk: Any) -> List[NormalizedToolCallItem]:
        """
        Extract tool calls from Gemini streaming chunk.
        
        Args:
            chunk: Gemini streaming event
            
        Returns:
            List of normalized tool call items
        """
        try:
            return self._extract_tool_calls_from_response(chunk)  # chunks use same shape
        except Exception:
            return []  # Ignore extraction errors

    def _create_tool_result_parts(self, tool_calls: List[NormalizedToolCallItem], results: List[Any]):
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
                parts.append(types.Part.from_function_response(name=tc["name"], response=response_data))
            except Exception:
                # Fallback: create a simple text part
                parts.append(types.Part(text=f"Function {tc['name']} returned: {res}"))
        return parts
