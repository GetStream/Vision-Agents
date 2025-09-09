from typing import Optional, List, TYPE_CHECKING, Any

from google import genai

from stream_agents.core.llm.llm import LLM, LLMResponse

from stream_agents.core.processors import BaseProcessor

if TYPE_CHECKING:
    from stream_agents.core.agents.conversation import Message


class GeminiLLM(LLM):
    """
    Use the SDK to keep history. (which is partially manual)
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        client: Optional[genai.Client] = None,
    ):
        super().__init__()
        self.model = model
        self.chat: Optional[Any] = None

        if client is not None:
            self.client = client
        else:
            self.client = genai.Client(api_key=api_key)

    async def simple_response(
        self,
        text: str,
        processors: Optional[List[BaseProcessor]] = None,
        participant: Any = None,
    ) -> LLMResponse:
        return await self.send_message(message=text)

    # basically wrap the Gemini native endpoint
    async def send_message(self, *args: Any, **kwargs: Any) -> LLMResponse:
        # if "model" not in kwargs:
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
    def _normalize_message(gemini_input) -> List["Message"]:
        from stream_agents.core.agents.conversation import Message

        # standardize on input
        if isinstance(gemini_input, str):
            gemini_input = [gemini_input]

        if not isinstance(gemini_input, List):
            gemini_input = [gemini_input]

        messages = []
        for i in gemini_input:
            message = Message(original=i, content=i)
            messages.append(message)

        return messages
