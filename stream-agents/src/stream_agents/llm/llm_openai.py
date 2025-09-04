from typing import Any, List, Optional, Dict
from openai import AsyncOpenAI

from stream_agents.processors import BaseProcessor

from .llm import LLM


class OpenAILLM(LLM):
    '''
    The goal is to standardize the minimal feature set thats needed for the agent integration
    That means

    - sharing instructions
    - keeping conversation history
    - response normalization

    Other than that we aim to give access to the native API methods from openAI as much as possible
    '''
    def __init__(self, model: str, api_key: Optional[str] = None, client: Optional[AsyncOpenAI] = None):
        if client is not None:
            self.client = client
        else:
            self.client = AsyncOpenAI()
        super().__init__(provider_name="openai", model=model, client=client)

    async def _create_response_impl(
        self,
        *,
        text: str,
        processors: Optional[List[BaseProcessor]] = None,
    ) -> str:
        req_input = text

        # Call OpenAI Responses API
        resp = await self.client.responses.create(model=self.model, input=req_input)

        # Try best-effort text extraction (Responses API)
        # Prefer output_text if available
        text = getattr(resp, "output_text", None)
        if isinstance(text, str) and text:
            return text

        # Fallback: traverse output items
        output = getattr(resp, "output", None) or []
        for item in output:
            if getattr(item, "type", None) == "message":
                parts = getattr(item, "content", None) or []
                for part in parts:
                    if getattr(part, "type", None) == "output_text":
                        val = getattr(part, "text", None)
                        if isinstance(val, str) and val:
                            return val

        # Last fallback: string cast of response
        return str(resp)