from typing import Optional, List, ParamSpec, TypeVar, TYPE_CHECKING, Any, Dict
import json

from openai import AsyncOpenAI
from openai.lib.streaming.responses import ResponseStreamEvent
from openai.types.responses import ResponseCompletedEvent, ResponseTextDeltaEvent, Response as OpenAIResponse

from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant

from vision_agents.core.llm.llm import LLM, LLMResponseEvent
from vision_agents.core.llm.llm_types import ToolSchema, NormalizedToolCallItem
from vision_agents.core.llm.events import LLMResponseChunkEvent, LLMResponseCompletedEvent
from . import events

from vision_agents.core.processors import Processor

if TYPE_CHECKING:
    from vision_agents.core.agents.conversation import Message

P = ParamSpec("P")
R = TypeVar("R")


class OpenAILLM(LLM):
    """
    The OpenAILLM class provides full/native access to the openAI SDK methods.
    It only standardized the minimal feature set that's needed for the agent integration.

    The agent requires that we standardize:
    - sharing instructions
    - keeping conversation history
    - response normalization

    Notes on the OpenAI integration
    - the native method is called create_response (maps 1-1 to responses.create)
    - history is maintained using conversation.create()

    Examples:

        from vision_agents.plugins import openai
        llm = openai.LLM(model="gpt-5")

    """

    def __init__(self, model: str, api_key: Optional[str] = None, client: Optional[AsyncOpenAI] = None):
        """
        Initialize the OpenAILLM class.

        Args:
            model (str): The OpenAI model to use. https://platform.openai.com/docs/models
            api_key: optional API key. by default loads from OPENAI_API_KEY
            client: optional OpenAI client. by default creates a new client object.
        """
        super().__init__()
        self.events.register_events_from_module(events)
        self.model = model
        self.openai_conversation: Optional[Any] = None
        self.conversation = None

        if client is not None:
            self.client = client
        elif api_key is not None and api_key != "":
            self.client = AsyncOpenAI(api_key=api_key)
        else:
            self.client = AsyncOpenAI()

    async def simple_response(self, text: str, processors: Optional[List[Processor]] = None,
                              participant: Participant = None):
        """
        simple_response is a standardized way (across openai, claude, gemini etc.) to create a response.

        Args:
            text: The text to respond to
            processors: list of processors (which contain state) about the video/voice AI
            participant: optionally the participant object

        Examples:

            llm.simple_response("say hi to the user, be mean")
        """
        # Use enhanced instructions if available (includes markdown file contents)
        instructions = None
        if hasattr(self, 'parsed_instructions') and self.parsed_instructions:
            instructions = self._build_enhanced_instructions()
        elif self.conversation is not None:
            instructions = self.conversation.instructions

        return await self.create_response(
            input=text,
            instructions=instructions,
        )

    async def create_response(self, *args: Any, **kwargs: Any) -> LLMResponseEvent[OpenAIResponse]:
        """
        create_response gives you full support/access to the native openAI responses.create method
        this method wraps the openAI method and ensures we broadcast an event which the agent class hooks into
        """
        if "model" not in kwargs:
            kwargs["model"] = self.model
        if "stream" not in kwargs:
            kwargs["stream"] = True

        if not self.openai_conversation:
            self.openai_conversation = await self.client.conversations.create()
        kwargs["conversation"] = self.openai_conversation.id

        # Add tools if available - convert to OpenAI format
        tools_spec = self._get_tools_for_provider()
        if tools_spec:
            kwargs["tools"] = self._convert_tools_to_provider_format(tools_spec)  # type: ignore[arg-type]

        # Use parsed instructions if available (includes markdown file contents)
        if hasattr(self, 'parsed_instructions') and self.parsed_instructions:

            # Combine original instructions with markdown file contents
            enhanced_instructions = self._build_enhanced_instructions()
            if enhanced_instructions:
                kwargs["instructions"] = enhanced_instructions

        # Set up input parameter for OpenAI Responses API
        if "input" not in kwargs:
            # Use the first positional argument as input, or create a default
            input_content = args[0] if args else "Hello"
            kwargs["input"] = input_content
        
        # OpenAI Responses API only accepts keyword arguments
        response = await self.client.responses.create(**kwargs)

        llm_response : Optional[LLMResponseEvent[OpenAIResponse]] = None

        if isinstance(response, OpenAIResponse):
            # Non-streaming response
            llm_response = LLMResponseEvent[OpenAIResponse](response, response.output_text)
            
            # Check for tool calls in non-streaming response
            tool_calls = self._extract_tool_calls_from_response(response)
            if tool_calls:
                # Execute tools and get follow-up response
                llm_response = await self._handle_tool_calls(tool_calls, kwargs)
                
        elif hasattr(response, "__aiter__"):  # async stream
            # Streaming response
            stream_response = response
            pending_tool_calls = []
            seen = set()
            
            # Process streaming events and collect tool calls
            async for event in stream_response:
                llm_response_optional = self._standardize_and_emit_event(event)
                if llm_response_optional is not None:
                    llm_response = llm_response_optional
                
                # Grab tool calls when the model finalizes the turn
                if getattr(event, "type", "") == "response.completed":
                    calls = self._extract_tool_calls_from_response(event.response)
                    for c in calls:
                        key = (c["id"], c["name"], json.dumps(c["arguments_json"], sort_keys=True))
                        if key not in seen:
                            pending_tool_calls.append(c)
                            seen.add(key)
            
            # If we have tool calls, execute them and get follow-up response
            if pending_tool_calls:
                llm_response = await self._handle_tool_calls(pending_tool_calls, kwargs)
        else:
            # Defensive fallback for unknown response types
            llm_response = LLMResponseEvent[OpenAIResponse](None, "")  # type: ignore[arg-type]

        # Note: For streaming responses, LLMResponseCompletedEvent is already emitted
        # in _standardize_and_emit_event when processing "response.completed" event.
        # Only emit it here for non-streaming responses to avoid duplication.
        if llm_response is not None and isinstance(response, OpenAIResponse):
            # Non-streaming response - emit completion event
            self.events.send(LLMResponseCompletedEvent(
                original=llm_response.original,
                text=llm_response.text
            ))

        return llm_response or LLMResponseEvent[OpenAIResponse](None, "")  # type: ignore[arg-type]

    async def _handle_tool_calls(self, tool_calls: List[NormalizedToolCallItem], original_kwargs: Dict[str, Any]) -> LLMResponseEvent[OpenAIResponse]:
        """
        Handle tool calls by executing them and getting a follow-up response.
        Supports multi-round tool calling (max 3 rounds).
        
        Args:
            tool_calls: List of tool calls to execute
            original_kwargs: Original kwargs from the request
            
        Returns:
            LLM response with tool results
        """
        llm_response: Optional[LLMResponseEvent[OpenAIResponse]] = None
        max_rounds = 3
        current_tool_calls = tool_calls
        current_kwargs = original_kwargs.copy()
        seen: set[tuple] = set()
        
        for round_num in range(max_rounds):
            # Execute tools (with cross-round deduplication)
            triples, seen = await self._dedup_and_execute(current_tool_calls, max_concurrency=8, timeout_s=30, seen=seen)  # type: ignore[arg-type]
            
            # If no tools were executed, break the loop
            if not triples:
                break
            
            # Process all tool calls, including failed ones
            tool_messages = []
            for tc, res, err in triples:
                cid = tc.get("id")
                if not cid:
                    # Skip tool calls without ID - they can't be reported back
                    continue
                
                # Use error result if there was an error, otherwise use the result
                output = err if err is not None else res
                
                # Convert to string for OpenAI Responses API with sanitization
                output_str = self._sanitize_tool_output(output)
                tool_messages.append({
                    "type": "function_call_output",
                    "call_id": cid,
                    "output": output_str,
                })
            
            # Don't send empty tool result inputs
            if not tool_messages:
                return llm_response or LLMResponseEvent[OpenAIResponse](None, "")  # type: ignore[arg-type]
            
            # Send follow-up request with tool results
            if not self.openai_conversation:
                return llm_response or LLMResponseEvent[OpenAIResponse](None, "")  # type: ignore[arg-type]
            
            follow_up_kwargs = {
                "model": current_kwargs.get("model", self.model),
                "conversation": self.openai_conversation.id,
                "input": tool_messages,
                "stream": True,
            }
            
            # Include tools again for potential follow-up calls
            tools_spec = self._get_tools_for_provider()
            if tools_spec:
                follow_up_kwargs["tools"] = self._convert_tools_to_provider_format(tools_spec)  # type: ignore[arg-type]
            
            # Get follow-up response
            follow_up_response = await self.client.responses.create(**follow_up_kwargs)
            
            if isinstance(follow_up_response, OpenAIResponse):
                # Non-streaming response
                llm_response = LLMResponseEvent[OpenAIResponse](follow_up_response, follow_up_response.output_text)
                
                # Check for more tool calls
                next_tool_calls = self._extract_tool_calls_from_response(follow_up_response)
                if next_tool_calls and round_num < max_rounds - 1:
                    current_tool_calls = next_tool_calls
                    current_kwargs = follow_up_kwargs
                    continue
                else:
                    return llm_response
                    
            elif hasattr(follow_up_response, "__aiter__"):  # async stream
                stream_response = follow_up_response
                llm_response = None
                pending_tool_calls = []
                # Don't reset seen - keep deduplication across rounds
                
                async for event in stream_response:
                    llm_response_optional = self._standardize_and_emit_event(event)
                    if llm_response_optional is not None:
                        llm_response = llm_response_optional
                    
                    # Check for more tool calls
                    if getattr(event, "type", "") == "response.completed":
                        calls = self._extract_tool_calls_from_response(event.response)
                        for c in calls:
                            key = (c["id"], c["name"], json.dumps(c["arguments_json"], sort_keys=True))
                            if key not in seen:
                                pending_tool_calls.append(c)
                                seen.add(key)
                
                # If we have more tool calls and haven't exceeded max rounds, continue
                if pending_tool_calls and round_num < max_rounds - 1:
                    current_tool_calls = pending_tool_calls
                    current_kwargs = follow_up_kwargs
                    continue
                else:
                    return llm_response or LLMResponseEvent[OpenAIResponse](None, "")  # type: ignore[arg-type]
            else:
                # Defensive fallback
                return LLMResponseEvent[OpenAIResponse](None, "")  # type: ignore[arg-type]
        
        # If we've exhausted all rounds, return the last response
        return llm_response or LLMResponseEvent[OpenAIResponse](None, "")  # type: ignore[arg-type]

    @staticmethod
    def _normalize_message(openai_input) -> List["Message"]:
        """
        Takes the openAI list of messages and standardizes it so we can store it in chat
        """
        from vision_agents.core.agents.conversation import Message

        # standardize on input
        if isinstance(openai_input, str):
            openai_input = [
                dict(content=openai_input, role="user", type="message")
            ]
        elif not isinstance(openai_input, List):
            openai_input = [openai_input]

        messages: List[Message] = []
        for i in openai_input:
            content = i.get("content", i if isinstance(i, str) else json.dumps(i))
            message = Message(original=i, content=content)
            messages.append(message)

        return messages



    def _convert_tools_to_provider_format(self, tools: List[ToolSchema]) -> List[Dict[str, Any]]:
        """
        Convert ToolSchema objects to OpenAI Responses API format.
        
        Args:
            tools: List of ToolSchema objects from the function registry
            
        Returns:
            List of tools in OpenAI Responses API format
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
                "name": name,                 # <-- top-level
                "description": description,   # <-- top-level
                "parameters": params,         # <-- top-level
                "strict": True,               # optional but fine
            })
        return out

    def _extract_tool_calls_from_response(self, response: Any) -> List[NormalizedToolCallItem]:
        """
        Extract tool calls from OpenAI response.
        
        Args:
            response: OpenAI response object
            
        Returns:
            List of normalized tool call items
        """
        calls = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "function_call":
                args = getattr(item, "arguments", "{}")
                try:
                    args_obj = json.loads(args) if isinstance(args, str) else args
                except Exception:
                    args_obj = {}
                call_item: NormalizedToolCallItem = {
                    "type": "tool_call",
                    "id": getattr(item, "call_id", ""),        # <-- call_id
                    "name": getattr(item, "name", "unknown"),
                    "arguments_json": args_obj,
                }
                calls.append(call_item)
        return calls


    def _create_tool_result_message(self, tool_calls: List[NormalizedToolCallItem], results: List[Any]) -> List[Dict[str, Any]]:
        """
        Create tool result messages for OpenAI Responses API.
        
        Args:
            tool_calls: List of tool calls that were executed
            results: List of results from function execution
            
        Returns:
            List of tool result messages in Responses API format
        """
        msgs = []
        for tc, res in zip(tool_calls, results):
            call_id = tc.get("id")
            if not call_id:
                # skip or wrap into a normal assistant message / log an error
                continue
                
            # Send only function_call_output items keyed by call_id
            # Convert to string for Responses API
            output_str = res if isinstance(res, str) else json.dumps(res)
            msgs.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": output_str,
            })
        return msgs

    def _standardize_and_emit_event(self, event: ResponseStreamEvent) -> Optional[LLMResponseEvent]:
        """
        Forwards the events and also send out a standardized version (the agent class hooks into that)
        """
        # start by forwarding the native event
        self.events.send(events.OpenAIStreamEvent(
            plugin_name="openai",
            event_type=event.type,
            event_data=event
        ))

        if event.type == "response.error":
            # Handle error events
            error_message = getattr(event, "error", {}).get("message", "Unknown error")
            self.events.send(events.LLMErrorEvent(
                plugin_name="openai",
                error_message=error_message,
                event_data=event
            ))
            return None
        elif event.type == "response.output_text.delta":
            # standardize the delta event
            delta_event: ResponseTextDeltaEvent = event
            self.events.send(LLMResponseChunkEvent(
                plugin_name="openai",
                content_index=delta_event.content_index,
                item_id=delta_event.item_id,
                output_index=delta_event.output_index,
                sequence_number=delta_event.sequence_number,
                delta=delta_event.delta,
            ))
        elif event.type == "response.completed":
            # standardize the response event and return the llm response
            completed_event: ResponseCompletedEvent = event
            llm_response = LLMResponseEvent[OpenAIResponse](completed_event.response, completed_event.response.output_text)
            self.events.send(LLMResponseCompletedEvent(
                plugin_name="openai",
                original=llm_response.original,
                text=llm_response.text
            ))
            return llm_response
        return None
