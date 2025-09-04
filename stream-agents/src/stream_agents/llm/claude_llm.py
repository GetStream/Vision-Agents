from typing import Optional, List

from anthropic import AsyncAnthropic

from stream_agents.llm.llm import LLM, LLMResponse
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
            # If no api_key provided, AsyncAnthropic will look for ANTHROPIC_API_KEY env var
            self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def create_message(self, *args, **kwargs) -> LLMResponse:
        # TODO: store message history here
        if "model" not in kwargs:
            kwargs["model"] = self.model

        original = await self.client.messages.create(*args, **kwargs)
        # TODO: update message history here with response
        return LLMResponse(original)

    async def simple_response(self, text: str, processors: Optional[List[BaseProcessor]] = None):
        return await self.create_message(
            input=text
        )