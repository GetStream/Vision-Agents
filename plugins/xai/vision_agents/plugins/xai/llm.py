import json
from typing import Optional, List, Any, TYPE_CHECKING, Dict
from xai_sdk import AsyncClient
from xai_sdk.chat import system, user, Response, Chunk, tool_result, tool
from xai_sdk.proto import chat_pb2

from vision_agents.core.llm.llm import LLM, LLMResponseEvent
from vision_agents.core.processors import Processor
from vision_agents.core.llm.events import (
    LLMResponseChunkEvent,
    LLMResponseCompletedEvent,
)
from vision_agents.core.llm.llm_types import NormalizedToolCallItem, ToolSchema
from . import events

if TYPE_CHECKING:
    from vision_agents.core.agents.conversation import Message
    from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant
    from xai_sdk.aio.chat import Chat
else:
    from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant


class XAILLM(LLM):
    """
    The XAILLM class provides full/native access to the xAI SDK methods.
    It only standardizes the minimal feature set that's needed for the agent integration.

    The agent requires that we standardize:
    - sharing instructions
    - keeping conversation history
    - response normalization

    Notes on the xAI integration
    - the native method is called create_response (maps to xAI chat.sample())
    - history is maintained using the chat object's append method

    Examples:

        from vision_agents.plugins import xai
        llm = xai.LLM(model="grok-4-latest")

    """

    def __init__(
        self,
        model: str = "grok-4-latest",
        api_key: Optional[str] = None,
        client: Optional[AsyncClient] = None,
    ):
        """
        Initialize the XAILLM class.

        Args:
            model (str): The xAI model to use. Defaults to "grok-4-latest"
            api_key: optional API key. by default loads from XAI_API_KEY
            client: optional xAI client. by default creates a new client object.
        """
        super().__init__()
        self.events.register_events_from_module(events)
        self.model = model
        self.xai_chat: Optional["Chat"] = None
        self.conversation = None

        if client is not None:
            self.client = client
        elif api_key is not None and api_key != "":
            self.client = AsyncClient(api_key=api_key)
        else:
            self.client = AsyncClient()

    async def simple_response(
        self,
        text: str,
        processors: Optional[List[Processor]] = None,
        participant: Optional[Participant] = None,
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
        instructions = None
        if self.conversation is not None:
            instructions = self.conversation.instructions

        return await self.create_response(
            input=text,
            instructions=instructions,
        )

    async def create_response(
        self, *args: Any, **kwargs: Any
    ) -> LLMResponseEvent[Response]:
        """
        create_response gives you full support/access to the native xAI chat.sample() and chat.stream() methods
        this method wraps the xAI method and ensures we broadcast an event which the agent class hooks into
        """
        input_text = kwargs.get("input", "")
        instructions = kwargs.get("instructions", "")
        model = kwargs.get("model", self.model)
        stream = kwargs.get("stream", True)

        # Get tools if available
        tools = self._get_tools_for_provider()

        # Create or reuse chat session
        if not self.xai_chat:
            messages = []
            if instructions:
                messages.append(system(instructions))
            create_kwargs = {"model": model, "messages": messages}
            if tools:
                create_kwargs["tools"] = tools
            self.xai_chat = self.client.chat.create(**create_kwargs)

        # Add user message
        assert self.xai_chat is not None
        self.xai_chat.append(user(input_text))

        # Get response based on streaming preference
        if stream:
            # Handle streaming response
            llm_response: Optional[LLMResponseEvent[Response]] = None
            pending_tool_calls = []
            seen = set()
            assert self.xai_chat is not None
            async for response, chunk in self.xai_chat.stream():
                llm_response_optional = self._standardize_and_emit_chunk(
                    chunk, response
                )
                if llm_response_optional is not None:
                    llm_response = llm_response_optional

                # Collect tool calls during streaming
                if chunk.choices and chunk.choices[0].finish_reason:
                    calls = self._extract_tool_calls_from_response(response)
                    for c in calls:
                        key = (
                            c.get("id"),
                            c["name"],
                            json.dumps(c.get("arguments_json", {}), sort_keys=True),
                        )
                        if key not in seen:
                            pending_tool_calls.append(c)
                            seen.add(key)

            # Add response to chat history
            if llm_response and llm_response.original:
                assert self.xai_chat is not None
                self.xai_chat.append(llm_response.original)

            # Handle tool calls if any
            if pending_tool_calls:
                llm_response = await self._handle_tool_calls(pending_tool_calls, kwargs)
        else:
            # Handle non-streaming response
            assert self.xai_chat is not None
            response = await self.xai_chat.sample()
            llm_response = LLMResponseEvent[Response](response, response.content)

            # Add response to chat history
            assert self.xai_chat is not None
            self.xai_chat.append(response)

            # Check for tool calls
            tool_calls = self._extract_tool_calls_from_response(response)
            if tool_calls:
                llm_response = await self._handle_tool_calls(tool_calls, kwargs)

        if llm_response is not None:
            self.events.send(
                LLMResponseCompletedEvent(
                    original=llm_response.original, text=llm_response.text
                )
            )

        return llm_response or LLMResponseEvent[Response](
            Response(chat_pb2.GetChatCompletionResponse(), 0), ""
        )

    @staticmethod
    def _normalize_message(input_text: str) -> List["Message"]:
        """
        Takes the input text and standardizes it so we can store it in chat
        """
        from vision_agents.core.agents.conversation import Message

        # Create a standardized message from input text
        message = Message(
            original={"content": input_text, "role": "user", "type": "message"},
            content=input_text,
        )

        return [message]

    def _convert_tools_to_provider_format(self, tools: List[ToolSchema]) -> List[Any]:
        """
        Convert ToolSchema objects to xAI SDK format.

        Args:
            tools: List of ToolSchema objects from the function registry

        Returns:
            List of tool objects in xAI SDK format
        """
        out = []
        for t in tools or []:
            if not isinstance(t, dict):
                continue
            name = t.get("name", "unnamed_tool")
            description = t.get("description", "") or ""
            params = t.get("parameters_schema") or t.get("parameters") or {}
            if not isinstance(params, dict):
                params = {}
            params.setdefault("type", "object")
            params.setdefault("properties", {})
            params.setdefault("additionalProperties", False)

            out.append(
                tool(
                    name=name,
                    description=description,
                    parameters=params,
                )
            )
        return out

    def _extract_tool_calls_from_response(
        self, response: Response
    ) -> List[NormalizedToolCallItem]:
        """
        Extract tool calls from xAI response.

        Args:
            response: xAI Response object

        Returns:
            List of normalized tool call items
        """
        calls = []
        tool_calls = getattr(response, "tool_calls", None) or []
        for tc in tool_calls:
            func = getattr(tc, "function", None)
            if not func:
                continue

            name = getattr(func, "name", "unknown")
            args_str = getattr(func, "arguments", "{}")
            call_id = getattr(tc, "id", "") or getattr(tc, "call_id", "")

            try:
                args_obj = (
                    json.loads(args_str) if isinstance(args_str, str) else args_str
                )
            except Exception:
                args_obj = {}

            call_item: NormalizedToolCallItem = {
                "type": "tool_call",
                "id": call_id,
                "name": name,
                "arguments_json": args_obj,
            }
            calls.append(call_item)
        return calls

    def _create_tool_result_message(
        self, tool_calls: List[NormalizedToolCallItem], results: List[Any]
    ) -> List[Any]:
        """
        Create tool result messages for xAI SDK.

        Args:
            tool_calls: List of tool calls that were executed
            results: List of results from function execution

        Returns:
            List of tool result messages in xAI SDK format
        """
        msgs = []
        for tc, res in zip(tool_calls, results):
            call_id = tc.get("id")
            if not call_id:
                continue

            output = res if isinstance(res, str) else json.dumps(res)
            output_str = self._sanitize_tool_output(output)
            msgs.append(tool_result(output_str))
        return msgs

    async def _handle_tool_calls(
        self, tool_calls: List[NormalizedToolCallItem], original_kwargs: Dict[str, Any]
    ) -> LLMResponseEvent[Response]:
        """
        Handle tool calls by executing them and getting a follow-up response.
        Supports multi-round tool calling (max 3 rounds).

        Args:
            tool_calls: List of tool calls to execute
            original_kwargs: Original kwargs from the request

        Returns:
            LLM response with tool results
        """
        llm_response: Optional[LLMResponseEvent[Response]] = None
        max_rounds = 3
        current_tool_calls = tool_calls
        seen: set[tuple] = set()

        for round_num in range(max_rounds):
            triples, seen = await self._dedup_and_execute(
                current_tool_calls,
                max_concurrency=8,
                timeout_s=30,
                seen=seen,
            )

            if not triples:
                break

            tool_results = []
            for tc, res, err in triples:
                cid = tc.get("id")
                if not cid:
                    continue

                output = err if err is not None else res
                output_str = self._sanitize_tool_output(output)
                tool_results.append(tool_result(output_str))

            if not tool_results:
                return llm_response or LLMResponseEvent[Response](
                    Response(chat_pb2.GetChatCompletionResponse(), 0), ""
                )

            if not self.xai_chat:
                return llm_response or LLMResponseEvent[Response](
                    Response(chat_pb2.GetChatCompletionResponse(), 0), ""
                )

            for tr in tool_results:
                self.xai_chat.append(tr)

            stream = original_kwargs.get("stream", True)
            if stream:
                llm_response = None
                pending_tool_calls = []

                async for response, chunk in self.xai_chat.stream():
                    llm_response_optional = self._standardize_and_emit_chunk(
                        chunk, response
                    )
                    if llm_response_optional is not None:
                        llm_response = llm_response_optional

                    if chunk.choices and chunk.choices[0].finish_reason:
                        calls = self._extract_tool_calls_from_response(response)
                        for c in calls:
                            key = (
                                c.get("id"),
                                c["name"],
                                json.dumps(c.get("arguments_json", {}), sort_keys=True),
                            )
                            if key not in seen:
                                pending_tool_calls.append(c)
                                seen.add(key)

                if llm_response and llm_response.original:
                    self.xai_chat.append(llm_response.original)

                if pending_tool_calls and round_num < max_rounds - 1:
                    current_tool_calls = pending_tool_calls
                    continue
                else:
                    return llm_response or LLMResponseEvent[Response](
                        Response(chat_pb2.GetChatCompletionResponse(), 0), ""
                    )
            else:
                response = await self.xai_chat.sample()
                llm_response = LLMResponseEvent[Response](response, response.content)
                self.xai_chat.append(response)

                next_tool_calls = self._extract_tool_calls_from_response(response)
                if next_tool_calls and round_num < max_rounds - 1:
                    current_tool_calls = next_tool_calls
                    continue
                else:
                    return llm_response

        return llm_response or LLMResponseEvent[Response](
            Response(chat_pb2.GetChatCompletionResponse(), 0), ""
        )

    def _standardize_and_emit_chunk(
        self, chunk: Chunk, response: Response
    ) -> Optional[LLMResponseEvent[Response]]:
        """
        Forwards the chunk events and also send out a standardized version (the agent class hooks into that)
        """
        # Emit the raw chunk event
        self.events.send(events.XAIChunkEvent(plugin_name="xai", chunk=chunk))

        # Emit standardized delta events for content
        if chunk.content:
            self.events.send(
                LLMResponseChunkEvent(
                    content_index=0,  # xAI doesn't have content_index
                    item_id=chunk.proto.id if hasattr(chunk.proto, "id") else "",
                    output_index=0,  # xAI doesn't have output_index
                    sequence_number=0,  # xAI doesn't have sequence_number
                    delta=chunk.content,
                    plugin_name="xai",
                )
            )

        # Check if this is the final chunk (finish_reason indicates completion)
        if chunk.choices and chunk.choices[0].finish_reason:
            # This is the final chunk, return the complete response
            llm_response = LLMResponseEvent[Response](response, response.content)
            self.events.send(
                LLMResponseCompletedEvent(
                    plugin_name="xai",
                    text=llm_response.text,
                    original=llm_response.original,
                )
            )
            return llm_response

        return None
