from typing import Optional, List, Iterable

import anthropic
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam

from stream_agents.llm.llm import LLM, LLMResponse

from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant
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

        if hasattr(self, "before_response_listener"):
            messages = self._normalize_input(kwargs["messages"])
            # TODO: figure this out
            #for m in messages:
            #    self.conversation.add_message(m["content"], "missing")
            self.before_response_listener(messages)
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
    def _normalize_input(messages: Iterable[MessageParam]):
        if isinstance(messages, str):
            messages = [{"content": messages, "role": "user", "type": "text"}]

        return messages