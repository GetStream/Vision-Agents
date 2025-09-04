from typing import Optional, List

from google import genai

from stream_agents.llm.llm import LLM, LLMResponse
from stream_agents.processors import BaseProcessor


class GeminiLLM(LLM):
    '''
    Use the SDK to keep history. (which is partially manual)
    '''
    def __init__(self, model: str, api_key: Optional[str] = None, client: Optional[genai.Client] = None):
        super().__init__()
        self.model = model
        self.chat = None

        if client is not None:
            self.client = client
        else:
            self.client = genai.Client(api_key=api_key)

    async def simple_response(self, text: str, processors: Optional[List[BaseProcessor]] = None):
        return await self.send_message(
            message=text
        )

    # basically wrap the Gemini native endpoint
    async def send_message(self, *args, **kwargs):
        #if "model" not in kwargs:
        #    kwargs["model"] = self.model

        # initialize chat if needed
        if self.chat is None:
            self.chat = self.client.chats.create(model=self.model)

        # Generate content using the client
        response = self.chat.send_message(*args, **kwargs)

        # Extract text from Gemini's response format
        text = response.text if response.text else ""
        return LLMResponse(response, text)