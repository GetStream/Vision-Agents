import datetime
from typing import Optional, List, Iterable, TYPE_CHECKING

import anthropic
from anthropic import AsyncAnthropic, AsyncStream, ContentBlockStopEvent, MessageStopEvent
from anthropic.types import MessageParam, RawMessageStreamEvent, Message as ClaudeMessage, RawMessageDeltaEvent, \
    RawContentBlockDeltaEvent, RawMessageStopEvent

from stream_agents.core.llm.llm import LLM, LLMResponse

from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant

from stream_agents.core.llm.types import StandardizedTextDeltaEvent
from stream_agents.core.processors import BaseProcessor

if TYPE_CHECKING:
    from stream_agents.core.agents.conversation import Message


class ClaudeLLM(LLM):
    '''
    Manually keep history
    '''
    def __init__(self, model: str, api_key: Optional[str] = None, client: Optional[AsyncAnthropic] = None):
        super().__init__()
        self.model = model

        if client is not None:
            self.client = client
        else:
            self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def create_message(self, *args, **kwargs) -> LLMResponse:
        if "model" not in kwargs:
            kwargs["model"] = self.model

        if "stream" not in kwargs:
            kwargs["stream"] = True

        # ensure the AI remembers the past conversation
        new_messages =  kwargs["messages"]
        old_messages = [m.original for m in self._conversation.messages]
        kwargs["messages"] = old_messages + new_messages

        self._conversation.add_messages(self._normalize_message(new_messages))

        if hasattr(self, "before_response_listener"):
            self.before_response_listener(new_messages)

        original = await self.client.messages.create(*args, **kwargs)
        if isinstance(original, ClaudeMessage):
            # Extract text from Claude's response format
            text = original.content[0].text if original.content else ""
            llm_response = LLMResponse(original, text)
        elif isinstance(original, AsyncStream):
            original : AsyncStream[RawMessageStreamEvent] = original
            total_text = ""
            async for event in original:
                if event.type=="content_block_delta":
                    print(event)
                    event: RawContentBlockDeltaEvent = event
                    total_text += event.delta.text

                    standardized_event = StandardizedTextDeltaEvent(
                        content_index=event.index,
                        type=event.type,
                        delta=event.delta.text,
                    )
                    self.emit("standardized.output_text.delta", standardized_event)
                    self.emit(event.type, event)
                elif event.type=="message_stop":
                    print(event)
                    event: RawMessageStopEvent = event
                    llm_response = LLMResponse(total_text, total_text)
                else:
                    self.emit(event.type, event)
                    pass

        if hasattr(self, "after_response_listener"):
            await self.after_response_listener(llm_response)


        return llm_response

    async def simple_response(self, text: str, processors: Optional[List[BaseProcessor]] = None, participant: Participant = None):
        return await self.create_message(
            messages=[{"role": "user", "content": text}],
            max_tokens=1000
        )

    @staticmethod
    def _normalize_message(claude_messages: Iterable["Message"]) -> List["Message"]:
        from stream_agents.core.agents.conversation import Message
        
        if isinstance(claude_messages, str):
            claude_messages = [{"content": claude_messages, "role": "user", "type": "text"}]

        if not isinstance(claude_messages, List):
            claude_messages = [claude_messages]

        messages : List[Message] = []
        for m in claude_messages:
            message = Message(original=m, content=m["content"], role=m["role"])
            messages.append(message)

        return messages