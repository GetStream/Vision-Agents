import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant
from openai import AsyncOpenAI
from openai.lib.streaming.responses import ResponseStreamEvent
from openai.types.responses import (
    Response as OpenAIResponse,
    ResponseCompletedEvent,
    ResponseTextDeltaEvent,
    ResponseFunctionToolCall,
)

from vision_agents.core.llm.events import (
    LLMResponseChunkEvent,
    LLMResponseCompletedEvent,
)
from vision_agents.core.llm.llm import LLM, LLMResponseEvent
from vision_agents.core.llm.llm_types import NormalizedToolCallItem, ToolSchema
from vision_agents.core.processors import Processor

from . import events
from .tool_utils import (
    convert_tools_to_openai_format,
    tool_call_dedup_key,
    parse_tool_arguments,
)

if TYPE_CHECKING:
    from vision_agents.core.agents.conversation import Message


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

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Optional[AsyncOpenAI] = None,
        max_tool_rounds: int = 3,
    ):
        """
        Initialize the OpenAILLM class.

        Args:
            model: The OpenAI model to use. https://platform.openai.com/docs/models
            api_key: Optional API key. By default loads from OPENAI_API_KEY.
            base_url: Optional base URL for the API.
            client: Optional OpenAI client. By default creates a new client object.
            max_tool_rounds: Maximum number of tool calling rounds (default 3).
        """
        super().__init__()
        self.events.register_events_from_module(events)
        self.model = model
        self.max_tool_rounds = max_tool_rounds
        self.openai_conversation: Optional[Any] = None
        self.conversation = None

        if client is not None:
            self.client = client
        elif api_key is not None and api_key != "":
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = AsyncOpenAI(base_url=base_url)

    async def simple_response(
        self,
        text: str,
        processors: Optional[List[Processor]] = None,
        participant: Participant = None,
    ):
        """
        simple_response is a standardized way (across openai, claude, gemini etc.) to create a response.

        Args:
            text: The text to respond to
            processors: list of processors (which contain state) about the video/voice AI
            participant: optionally the participant object

        Examples:

            llm.simple_response("say hi to the user, be mean")
        """

        return await self.create_response(
            input=text,
            instructions=self._instructions,
        )

    async def create_conversation(self):
        if not self.openai_conversation:
            self.openai_conversation = await self.client.conversations.create()

    def add_conversation_history(self, kwargs):
        if self.openai_conversation:
            kwargs["conversation"] = self.openai_conversation.id

    async def create_response(
        self, *args: Any, **kwargs: Any
    ) -> LLMResponseEvent[OpenAIResponse]:
        """
        create_response gives you full support/access to the native openAI responses.create method
        this method wraps the openAI method and ensures we broadcast an event which the agent class hooks into
        """
        if "model" not in kwargs:
            kwargs["model"] = self.model
        if "stream" not in kwargs:
            kwargs["stream"] = True

        # create the conversation if needed and add the required args
        await self.create_conversation()
        self.add_conversation_history(kwargs)

        # Add tools if available
        tools = self.get_available_functions()
        if tools:
            kwargs["tools"] = convert_tools_to_openai_format(tools)

        # Provide instructions
        if self._instructions:
            kwargs["instructions"] = self._instructions

        # Set up input parameter for OpenAI Responses API
        if "input" not in kwargs:
            # Use the first positional argument as input, or create a default
            input_content = args[0] if args else "Hello"
            kwargs["input"] = input_content

        # OpenAI Responses API only accepts keyword arguments
        response = await self.client.responses.create(**kwargs)

        llm_response: Optional[LLMResponseEvent[OpenAIResponse]] = None

        if isinstance(response, OpenAIResponse):
            # Non-streaming response
            llm_response = LLMResponseEvent[OpenAIResponse](
                response, response.output_text
            )

            # Check for tool calls in non-streaming response
            tool_calls = self._extract_tool_calls_from_response(response)
            if tool_calls:
                # Execute tools and get follow-up response
                llm_response = await self._handle_tool_calls(tool_calls, kwargs)

        elif hasattr(response, "__aiter__"):  # async stream
            # Streaming response
            stream_response = response
            pending_tool_calls: list[NormalizedToolCallItem] = []
            seen: set[tuple[str, str]] = set()

            # Process streaming events and collect tool calls
            async for event in stream_response:
                llm_response_optional = self._standardize_and_emit_event(event)
                if llm_response_optional is not None:
                    llm_response = llm_response_optional

                # Grab tool calls when the model finalizes the turn
                if event.type == "response.completed":
                    calls = self._extract_tool_calls_from_response(event.response)
                    for c in calls:
                        key = tool_call_dedup_key(c)
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
            self.events.send(
                LLMResponseCompletedEvent(
                    item_id=llm_response.original.output[0].id,
                    original=llm_response.original,
                    text=llm_response.text,
                )
            )

        return llm_response or LLMResponseEvent[OpenAIResponse](None, "")  # type: ignore[arg-type]

    async def _handle_tool_calls(
        self, tool_calls: List[NormalizedToolCallItem], original_kwargs: Dict[str, Any]
    ) -> LLMResponseEvent[OpenAIResponse]:
        """Execute tool calls and get follow-up response. Supports multi-round.

        Args:
            tool_calls: List of tool calls to execute
            original_kwargs: Original kwargs from the request

        Returns:
            LLM response with tool results
        """
        llm_response: Optional[LLMResponseEvent[OpenAIResponse]] = None
        current_tool_calls = tool_calls
        seen: set[tuple[str, str]] = set()

        for round_num in range(self.max_tool_rounds):
            # Execute tools with deduplication
            triples, seen = await self._dedup_and_execute(
                current_tool_calls,
                max_concurrency=8,
                timeout_s=30,
                seen=seen,
            )

            if not triples:
                break

            # Build tool result messages
            tool_messages = self._build_tool_messages(triples)
            if not tool_messages:
                break

            # Get follow-up response
            (
                llm_response,
                next_tool_calls,
            ) = await self._send_tool_results_and_get_response(tool_messages, seen)

            if not next_tool_calls or round_num >= self.max_tool_rounds - 1:
                break

            current_tool_calls = next_tool_calls

        return llm_response or LLMResponseEvent[OpenAIResponse](None, "")  # type: ignore[arg-type]

    def _build_tool_messages(
        self, triples: list[tuple[dict[str, Any], Any, Any]]
    ) -> list[dict[str, Any]]:
        """Build tool result messages from execution results."""
        tool_messages = []
        for tc, res, err in triples:
            call_id = tc.get("id")
            if not call_id:
                continue

            output = err if err is not None else res
            output_str = self._sanitize_tool_output(output)
            tool_messages.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output_str,
                }
            )
        return tool_messages

    async def _send_tool_results_and_get_response(
        self,
        tool_messages: list[dict[str, Any]],
        seen: set[tuple[str, str]],
    ) -> tuple[
        Optional[LLMResponseEvent[OpenAIResponse]], List[NormalizedToolCallItem]
    ]:
        """Send tool results and get follow-up response.

        Returns:
            Tuple of (llm_response, next_tool_calls)
        """
        if not self.openai_conversation:
            return None, []

        follow_up_kwargs: Dict[str, Any] = {
            "model": self.model,
            "conversation": self.openai_conversation.id,
            "input": tool_messages,
            "stream": True,
        }

        # Include tools for potential follow-up calls
        tools = self.get_available_functions()
        if tools:
            follow_up_kwargs["tools"] = convert_tools_to_openai_format(tools)

        follow_up_response = await self.client.responses.create(**follow_up_kwargs)

        if isinstance(follow_up_response, OpenAIResponse):
            llm_response = LLMResponseEvent[OpenAIResponse](
                follow_up_response, follow_up_response.output_text
            )
            next_tool_calls = self._extract_tool_calls_from_response(follow_up_response)
            return llm_response, next_tool_calls

        # Streaming response
        llm_response_streaming: Optional[LLMResponseEvent[OpenAIResponse]] = None
        pending_tool_calls: List[NormalizedToolCallItem] = []

        async for event in follow_up_response:
            llm_response_optional = self._standardize_and_emit_event(event)
            if llm_response_optional is not None:
                llm_response_streaming = llm_response_optional

            if event.type == "response.completed":
                calls = self._extract_tool_calls_from_response(event.response)
                for c in calls:
                    key = tool_call_dedup_key(c)
                    if key not in seen:
                        pending_tool_calls.append(c)
                        seen.add(key)

        return llm_response_streaming, pending_tool_calls

    @staticmethod
    def _normalize_message(openai_input) -> List["Message"]:
        """
        Takes the openAI list of messages and standardizes it so we can store it in chat
        """
        from vision_agents.core.agents.conversation import Message

        # standardize on input
        if isinstance(openai_input, str):
            openai_input = [dict(content=openai_input, role="user", type="message")]
        elif not isinstance(openai_input, List):
            openai_input = [openai_input]

        messages: List[Message] = []
        for i in openai_input:
            content = i.get("content", i if isinstance(i, str) else json.dumps(i))
            message = Message(original=i, content=content)
            messages.append(message)

        return messages

    def _convert_tools_to_provider_format(
        self, tools: List[ToolSchema]
    ) -> List[Dict[str, Any]]:
        """Convert ToolSchema objects to OpenAI format."""
        return convert_tools_to_openai_format(tools)

    def _extract_tool_calls_from_response(
        self, response: OpenAIResponse
    ) -> List[NormalizedToolCallItem]:
        """Extract tool calls from OpenAI response."""
        calls: List[NormalizedToolCallItem] = []
        for item in response.output or []:
            if isinstance(item, ResponseFunctionToolCall):
                calls.append(
                    {
                        "type": "tool_call",
                        "id": item.call_id,
                        "name": item.name,
                        "arguments_json": parse_tool_arguments(item.arguments),
                    }
                )
        return calls

    def _create_tool_result_message(
        self, tool_calls: List[NormalizedToolCallItem], results: List[Any]
    ) -> List[Dict[str, Any]]:
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
            msgs.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output_str,
                }
            )
        return msgs

    def _standardize_and_emit_event(
        self, event: ResponseStreamEvent
    ) -> Optional[LLMResponseEvent[OpenAIResponse]]:
        """Forward native events and emit standardized versions."""
        # Forward the native event
        self.events.send(
            events.OpenAIStreamEvent(
                plugin_name="openai", event_type=event.type, event_data=event
            )
        )

        if event.type == "response.error":
            error_message = event.error.message if event.error else "Unknown error"
            self.events.send(
                events.LLMErrorEvent(
                    plugin_name="openai", error_message=error_message, event_data=event
                )
            )
            return None

        if event.type == "response.output_text.delta":
            delta_event: ResponseTextDeltaEvent = event
            self.events.send(
                LLMResponseChunkEvent(
                    plugin_name="openai",
                    content_index=None,
                    item_id=delta_event.item_id,
                    output_index=delta_event.output_index,
                    sequence_number=delta_event.sequence_number,
                    delta=delta_event.delta,
                )
            )
            return None

        if event.type == "response.completed":
            completed_event: ResponseCompletedEvent = event
            llm_response = LLMResponseEvent[OpenAIResponse](
                completed_event.response, completed_event.response.output_text
            )
            self.events.send(
                LLMResponseCompletedEvent(
                    plugin_name="openai",
                    original=llm_response.original,
                    text=llm_response.text,
                    item_id=llm_response.original.output[0].id,
                )
            )
            return llm_response

        return None
