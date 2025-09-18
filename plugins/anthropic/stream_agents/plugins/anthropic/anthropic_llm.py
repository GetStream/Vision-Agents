from typing import Optional, List, TYPE_CHECKING, Any

import anthropic
from anthropic import AsyncAnthropic, AsyncStream
from anthropic.types import (
    RawMessageStreamEvent,
    Message as ClaudeMessage,
    RawContentBlockDeltaEvent,
    RawMessageStopEvent,
)

from stream_agents.core.llm.llm import LLM, LLMResponseEvent

from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant

from stream_agents.core.llm.types import StandardizedTextDeltaEvent
from stream_agents.core.processors import BaseProcessor

if TYPE_CHECKING:
    from stream_agents.core.agents.conversation import Message


class ClaudeLLM(LLM):
    """
    The ClaudeLLM class provides full/native access to the claude SDK methods.
    It only standardized the minimal feature set that's needed for the agent integration.

    The agent requires that we standardize:
    - sharing instructions
    - keeping conversation history
    - response normalization

    Notes on the Claude integration
    - the native method is called create_message (maps 1-1 to messages.create)
    - history is maintained manually by keeping it in memory

    Examples:

        from stream_agents.plugins import anthropic
        llm = anthropic.LLM(model="claude-opus-4-1-20250805")
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        client: Optional[AsyncAnthropic] = None,
    ):
        """
        Initialize the ClaudeLLM class.

        Args:
            model (str): The model to use. https://docs.anthropic.com/en/docs/about-claude/models/overview
            api_key: optional API key. by default loads from ANTHROPIC_API_KEY
            client: optional Anthropic client. by default creates a new client object.
        """
        super().__init__()
        self.model = model

        if client is not None:
            self.client = client
        else:
            self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def simple_response(
        self,
        text: str,
        processors: Optional[List[BaseProcessor]] = None,
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
        return await self.create_message(
            messages=[{"role": "user", "content": text}], max_tokens=1000
        )

    async def create_message(self, *args, **kwargs) -> LLMResponseEvent:
        """
        create_message gives you full support/access to the native Claude message.create method
        this method wraps the Claude method and ensures we broadcast an event which the agent class hooks into
        """
        if "model" not in kwargs:
            kwargs["model"] = self.model

        if "stream" not in kwargs:
            kwargs["stream"] = True

        # ensure the AI remembers the past conversation
        new_messages = kwargs["messages"]
        if self._conversation:
            old_messages = [m.original for m in self._conversation.messages]
            kwargs["messages"] = old_messages + new_messages
            # Add messages to conversation
            normalized_messages = self._normalize_message(new_messages)
            for msg in normalized_messages:
                self._conversation.messages.append(msg)

        class BeforeLLMResponseEventEvent:
            pass

        self.events.send(self._normalize_message(kwargs["messages"]))

        original = await self.client.messages.create(*args, **kwargs)
        if isinstance(original, ClaudeMessage):
            # Extract text from Claude's response format
            text = ""
            if original.content and len(original.content) > 0:
                content_block = original.content[0]
                if hasattr(content_block, "text"):
                    text = content_block.text
            llm_response = LLMResponseEvent(original, text)
        elif isinstance(original, AsyncStream):
            stream: AsyncStream[RawMessageStreamEvent] = original
            text_parts: List[str] = []
            async for event in stream:
                llm_response_optional = self._standardize_and_emit_event(
                    event, text_parts
                )
                if llm_response_optional is not None:
                    llm_response = llm_response_optional

        class AfterLLMResponseEventEvent:
            pass

        self.emit("after_llm_response", llm_response)

        return llm_response

    def _standardize_and_emit_event(
        self, event: RawMessageStreamEvent, text_parts: List[str]
    ) -> Optional[LLMResponseEvent]:
        """
        Forwards the events and also send out a standardized version (the agent class hooks into that)
        """
        # forward the native event
        self.emit("claude_event", event)

        # send a standardized version for delta and response
        if event.type == "content_block_delta":
            delta_event: RawContentBlockDeltaEvent = event
            if hasattr(delta_event.delta, "text") and delta_event.delta.text:
                text_parts.append(delta_event.delta.text)

                standardized_event = StandardizedTextDeltaEvent(
                    content_index=delta_event.index,
                    item_id="",
                    output_index=0,
                    sequence_number=0,
                    type="response.output_text.delta",
                    delta=delta_event.delta.text,
                )
                self.emit("standardized.output_text.delta", standardized_event)
        elif event.type == "message_stop":
            stop_event: RawMessageStopEvent = event
            total_text = "".join(text_parts)
            llm_response = LLMResponseEvent(stop_event, total_text)
            return llm_response
        return None

    @staticmethod
    def _normalize_message(claude_messages: Any) -> List["Message"]:
        from stream_agents.core.agents.conversation import Message

        if isinstance(claude_messages, str):
            claude_messages = [
                {"content": claude_messages, "role": "user", "type": "text"}
            ]

        if not isinstance(claude_messages, (List, tuple)):
            claude_messages = [claude_messages]

        messages: List[Message] = []
        for m in claude_messages:
            if isinstance(m, dict):
                content = m.get("content", "")
                role = m.get("role", "user")
            else:
                content = str(m)
                role = "user"
            message = Message(original=m, content=content, role=role)
            messages.append(message)

        return messages
