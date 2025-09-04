from typing import Optional, List

from google import genai

from stream_agents.llm.llm import LLM, LLMResponse
from stream_agents.processors import BaseProcessor


class GeminiLLM(LLM):
    '''
    Use the SDK to keep history. (which is partially manual)
    '''
    client: genai.Client

    def __init__(self, model: str, api_key: Optional[str] = None, client: Optional[genai.Client] = None):
        super().__init__()
        self.model = model

        if client is not None:
            self.client = client
        else:
            self.client = genai.Client()

    async def simple_response(self, text: str, processors: Optional[List[BaseProcessor]] = None):
        return await self.generate_content(
            contents=text
        )

    # basically wrap the Gemini native endpoint
    def generate_content(self, *args, **kwargs):
        if "model" not in kwargs:
            kwargs["model"] = self.model

        client = genai.Client()
        chat = client.chats.create(model="gemini-2.5-flash")

        response = chat.send_message("I have 2 dogs in my house.")

        response = self.client.generate_content(*args, **kwargs)

        return LLMResponse(response)