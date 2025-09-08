import datetime
from typing import Optional, List, Iterable

import anthropic
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam

from stream_agents.llm.llm import LLM, LLMResponse

from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant
from stream_agents.agents.conversation import Message
from stream_agents.processors import BaseProcessor


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

        # ensure the AI remembers the past conversation
        new_messages =  kwargs["messages"]
        old_messages = [m.original for m in self._conversation.messages]
        kwargs["messages"] = old_messages + new_messages

        self._conversation.add_messages(self._normalize_message(new_messages))

        if hasattr(self, "before_response_listener"):
            self.before_response_listener(new_messages)

        original = await self.client.messages.create(*args, **kwargs)

        # Extract text from Claude's response format
        text = original.content[0].text if original.content else ""
        llm_response = LLMResponse(original, text)
        if hasattr(self, "after_response_listener"):
            await self.after_response_listener(llm_response)
        return llm_response

    async def simple_response(self, text: str, processors: Optional[List[BaseProcessor]] = None, participant: Participant = None):
        return await self.create_message(
            messages=[{"role": "user", "content": text}],
            max_tokens=1000
        )

    @staticmethod
    def _normalize_message(claude_messages: Iterable[Message]) -> List[Message]:
        if isinstance(claude_messages, str):
            claude_messages = [{"content": claude_messages, "role": "user", "type": "text"}]

        if not isinstance(claude_messages, List):
            claude_messages = [claude_messages]

        messages : List[Message] = []
        for m in claude_messages:
            message = Message(original=m, content=m["content"], role=m["role"])
            messages.append(message)

        return messages