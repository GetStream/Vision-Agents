import uuid
from typing import Optional, List, TYPE_CHECKING, Any, Dict

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, ChatSession
    from vertexai.generative_models import GenerationConfig, Tool
    from google.auth.credentials import Credentials
except ImportError:
    vertexai = None
    GenerativeModel = None
    ChatSession = None
    GenerationConfig = None
    Tool = None
    Credentials = None

from vision_agents.core.llm.llm import LLM, LLMResponseEvent
from vision_agents.core.llm.llm_types import ToolSchema, NormalizedToolCallItem

from vision_agents.core.llm.events import LLMResponseCompletedEvent, LLMResponseChunkEvent

from . import events

from vision_agents.core.processors import Processor

if TYPE_CHECKING:
    from vision_agents.core.agents.conversation import Message


class VertexAILLM(LLM):
    """
    The VertexAILLM class provides full/native access to the Vertex AI SDK methods.
    It only standardizes the minimal feature set that's needed for the agent integration.

    The agent requires that we standardize:
    - sharing instructions
    - keeping conversation history
    - response normalization

    Notes on the Vertex AI integration:
    - the native method is called generate_content (maps 1-1 to chat.send_message with stream=True)
    - history is maintained in the Vertex AI SDK (with the usage of model.start_chat())
    - requires Google Cloud credentials (Application Default Credentials or service account)

    Examples:

        from vision_agents.plugins import vertex_ai
        llm = vertex_ai.LLM(model="gemini-1.5-pro", project="my-project", location="us-central1")
    """
    def __init__(
        self,
        model: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[Any] = None,
    ):
        """
        Initialize the VertexAILLM class.

        Args:
            model (str): The model to use (e.g., "gemini-1.5-pro", "gemini-1.5-flash").
            project: GCP project ID. Defaults to environment variable GCP_PROJECT or gcloud default.
            location: GCP location/region (e.g., "us-central1"). Defaults to environment variable GCP_LOCATION or "us-central1".
            credentials: Optional Google Cloud credentials. If None, uses Application Default Credentials.
        """
        super().__init__()
        self.events.register_events_from_module(events)
        self.model_name = model
        self.project = project
        self.location = location or "us-central1"
        self.chat: Optional[Any] = None
        self.vertex_model: Optional[GenerativeModel] = None

        # Initialize Vertex AI
        if vertexai is None:
            raise ImportError(
                "google-cloud-aiplatform is required. Install it with: pip install google-cloud-aiplatform"
            )

        # Initialize Vertex AI with project and location
        if credentials is not None:
            vertexai.init(project=project, location=self.location, credentials=credentials)
        else:
            vertexai.init(project=project, location=self.location)

        # Create the generative model (system_instruction and tools will be set when chat starts)
        # We'll recreate the model when system instructions change
        self.vertex_model = None  # Will be created on first use

    async def simple_response(
        self,
        text: str,
        processors: Optional[List[Processor]] = None,
        participant: Optional[Any] = None
    ) -> LLMResponseEvent[Any]:
        """
        simple_response is a standardized way (across openai, claude, gemini etc.) to create a response.

        Args:
            text: The text to respond to
            processors: list of processors (which contain state) about the video/voice AI

        Examples:

            llm.simple_response("say hi to the user, be mean")
        """
        return await self.generate_content(message=text)

    async def generate_content(self, *args, **kwargs):
        """
        generate_content gives you full support/access to the native Vertex AI generate_content method
        under the hood it calls chat.send_message(*args, stream=True, **kwargs)
        this method wraps and ensures we broadcast an event which the agent class hooks into
        """
        # Build system instructions if available
        enhanced_instructions = self._build_enhanced_instructions()
        
        # Get tools if available
        tools_spec = self.get_available_functions()
        conv_tools = None
        if tools_spec:
            conv_tools = self._convert_tools_to_provider_format(tools_spec)

        # Check if we need to recreate the model (if system instructions changed)
        current_instructions = getattr(self, '_current_instructions', None)
        if enhanced_instructions != current_instructions or self.vertex_model is None:
            # Create/update the generative model with system instruction and tools
            model_kwargs = {}
            if enhanced_instructions:
                model_kwargs['system_instruction'] = enhanced_instructions
            if conv_tools:
                model_kwargs['tools'] = conv_tools
            self.vertex_model = GenerativeModel(model_name=self.model_name, **model_kwargs)
            self._current_instructions = enhanced_instructions
            # Reset chat when model changes
            self.chat = None

        # Initialize chat if needed
        if self.chat is None:
            self.chat = self.vertex_model.start_chat()

        # Generate content using streaming
        message = kwargs.pop("message", args[0] if args else None)
        if message is None:
            raise ValueError("message is required")

        # Send message and get streaming response
        # Vertex AI ChatSession.send_message() returns an iterable when stream=True
        # If tools changed since model creation, pass them here
        send_kwargs = {"stream": True}
        if conv_tools:
            send_kwargs["tools"] = conv_tools
        response_stream = self.chat.send_message(message, **send_kwargs)
        
        text_parts: List[str] = []
        final_chunk = None
        pending_calls: List[NormalizedToolCallItem] = []

        # Vertex AI doesn't have an item_id in responses, we create it here
        item_id = str(uuid.uuid4())

        for idx, chunk in enumerate(response_stream):
            final_chunk = chunk
            self._standardize_and_emit_event(chunk, text_parts, item_id, idx)

            # Collect function calls as they stream
            try:
                chunk_calls = self._extract_tool_calls_from_stream_chunk(chunk)
                pending_calls.extend(chunk_calls)
            except Exception:
                pass  # Ignore errors in chunk processing

        # Check if there were function calls in the response
        if pending_calls:
            # Multi-hop tool calling loop
            MAX_ROUNDS = 3
            rounds = 0
            current_calls = pending_calls
            
            seen: set[str] = set()
            while current_calls and rounds < MAX_ROUNDS:
                # Execute tools concurrently with deduplication
                triples, seen = await self._dedup_and_execute(
                    current_calls, max_concurrency=8, timeout_s=30, seen=seen
                )
                
                # Create function response parts for Vertex AI
                function_response_parts = self._create_tool_result_parts(
                    [tc for tc, _, _ in triples],
                    [res for _, res, _ in triples]
                )
                
                # Send function responses back to the model
                follow_up_kwargs = {"stream": True}
                if conv_tools:
                    follow_up_kwargs["tools"] = conv_tools
                follow_up_stream = self.chat.send_message(function_response_parts, **follow_up_kwargs)
                
                follow_up_text_parts: List[str] = []
                follow_up_last = None
                next_calls = []
                
                for idx, chk in enumerate(follow_up_stream):
                    follow_up_last = chk
                    self._standardize_and_emit_event(chk, follow_up_text_parts, item_id, idx)

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
            plugin_name="vertex_ai",
            original=llm_response.original,
            text=llm_response.text,
            item_id=item_id,
        ))

        # Return the LLM response
        return llm_response

    @staticmethod
    def _normalize_message(vertex_input) -> List["Message"]:
        from vision_agents.core.agents.conversation import Message
        
        # Standardize on input
        if isinstance(vertex_input, str):
            vertex_input = [vertex_input]

        if not isinstance(vertex_input, List):
            vertex_input = [vertex_input]

        messages = []
        for i in vertex_input:
            message = Message(original=i, content=i)
            messages.append(message)

        return messages

    def _standardize_and_emit_event(
        self,
        chunk: Any,
        text_parts: List[str],
        item_id: str,
        idx: int
    ) -> Optional[LLMResponseEvent[Any]]:
        """
        Forwards the events and also send out a standardized version (the agent class hooks into that)
        """
        # Forward the native event
        self.events.send(events.VertexAIResponseEvent(
            plugin_name="vertex_ai",
            response_chunk=chunk
        ))

        # Check if response has text content
        # Vertex AI chunks have a text property
        chunk_text = None
        if hasattr(chunk, 'text'):
            chunk_text = chunk.text
        elif hasattr(chunk, 'candidates') and chunk.candidates:
            # Try to get text from candidates
            candidate = chunk.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        if hasattr(part, 'text'):
                            chunk_text = part.text
                            break

        if chunk_text:
            self.events.send(LLMResponseChunkEvent(
                plugin_name="vertex_ai",
                content_index=idx,
                item_id=item_id,
                delta=chunk_text,
            ))
            text_parts.append(chunk_text)

        return None

    def _convert_tools_to_provider_format(self, tools: List[ToolSchema]) -> List[Tool]:
        """
        Convert ToolSchema objects to Vertex AI format.
        Args:
            tools: List of ToolSchema objects
        Returns:
            List of Tool objects for Vertex AI
        """
        if Tool is None:
            return []
        
        function_declarations = []
        for tool in tools:
            function_declarations.append({
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool["parameters_schema"]
            })
        
        # Vertex AI expects a single Tool object containing function_declarations
        if function_declarations:
            return [Tool.from_function_declarations(function_declarations)]
        return []

    def _extract_tool_calls_from_response(self, response: Any) -> List[NormalizedToolCallItem]:
        """
        Extract tool calls from Vertex AI response.
        
        Args:
            response: Vertex AI response object
            
        Returns:
            List of normalized tool call items
        """
        calls: List[NormalizedToolCallItem] = []
        
        try:
            # Vertex AI response structure
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and candidate.content:
                        if hasattr(candidate.content, 'parts'):
                            for part in candidate.content.parts:
                                if hasattr(part, 'function_call'):
                                    fc = part.function_call
                                    calls.append({
                                        "type": "tool_call",
                                        "name": getattr(fc, "name", "unknown"),
                                        "arguments_json": getattr(fc, "args", {}),
                                    })
        except Exception:
            pass  # Ignore extraction errors
        
        return calls

    def _extract_tool_calls_from_stream_chunk(self, chunk: Any) -> List[NormalizedToolCallItem]:
        """
        Extract tool calls from Vertex AI streaming chunk.
        
        Args:
            chunk: Vertex AI streaming event
            
        Returns:
            List of normalized tool call items
        """
        try:
            return self._extract_tool_calls_from_response(chunk)  # chunks use same shape
        except Exception:
            return []  # Ignore extraction errors

    def _create_tool_result_parts(self, tool_calls: List[NormalizedToolCallItem], results: List[Any]):
        """
        Create function_response parts for Vertex AI.
        
        Args:
            tool_calls: List of tool calls that were executed
            results: List of results from function execution
            
        Returns:
            List of function_response parts
        """
        parts = []
        try:
            from vertexai.generative_models import FunctionResponse, Part
            
            for tc, res in zip(tool_calls, results):
                try:
                    # Convert result to dict if it's not already
                    if isinstance(res, dict):
                        response_data = res
                    else:
                        response_data = {"result": res}
                    
                    # Sanitize large outputs
                    sanitized_res = {}
                    for k, v in response_data.items():
                        sanitized_res[k] = self._sanitize_tool_output(v)
                    
                    # Create function response part
                    function_response = FunctionResponse(
                        name=tc["name"],
                        response=sanitized_res
                    )
                    parts.append(Part(function_response=function_response))
                except Exception:
                    # Fallback: create a simple text part
                    try:
                        parts.append(Part(text=f"Function {tc['name']} returned: {res}"))
                    except Exception:
                        pass  # Skip if Part creation fails
        except ImportError:
            # Fallback if import fails
            pass
        
        return parts
