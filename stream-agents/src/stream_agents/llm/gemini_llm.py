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

    @staticmethod
    def _normalize_message(openai_input) -> List[Message]:
        # standardize on input
        if isinstance(openai_input, str):
            openai_input = [
                dict(content=openai_input, role="user", type="message")
            ]
        elif not isinstance(openai_input, List):
            openai_input = [openai_input]

        messages = []
        for i in openai_input:
            message = Message(original=i)
            message.content = i["content"]
            messages.append(message)

        return messages