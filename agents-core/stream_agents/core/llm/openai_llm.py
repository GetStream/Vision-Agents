from typing import Optional, List, TypeVar, Callable, TYPE_CHECKING, Any, no_type_check

from openai import OpenAI

from getstream.models import Response
import inspect
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant
from stream_agents.core.llm.llm import LLM, LLMResponse

from stream_agents.core.processors import BaseProcessor

if TYPE_CHECKING:
    from stream_agents.core.agents.conversation import Message


R = TypeVar("R")


def use_create(fn: Callable[..., R]) -> Callable[..., R]:
    return fn


class OpenAILLM(LLM):
    """
    The goal is to expose the regular/native openAI SDK methods,
    and only standardize the minimal feature set that's needed for the agent integration.
    """

    def __init__(
        self, model: str, api_key: Optional[str] = None, client: Optional[OpenAI] = None
    ):
        super().__init__()
        self.model = model
        self.openai_conversation: Optional[Any] = None
        self.conversation = None

        if client is not None:
            self.client: OpenAI = client
        elif api_key is not None and api_key != "":
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI()

    @no_type_check
    async def create_response(self, *args: Any, **kwargs: Any) -> LLMResponse[Response]:
        if "model" not in kwargs:
            kwargs["model"] = self.model

        if self.openai_conversation is None:
            # Using beta conversations API
            self.openai_conversation = self.client.conversations.create()
        kwargs["conversation"] = getattr(self.openai_conversation, "id", None)

        if hasattr(self, "before_response_listener") and "input" in kwargs:
            self.before_response_listener(self._normalize_message(kwargs["input"]))
        response: Response = self.client.responses.create(*args, **kwargs)

        llm_response: LLMResponse[Response] = LLMResponse(
            response, response.output_text
        )
        if hasattr(self, "after_response_listener"):
            maybe_awaitable = self.after_response_listener(llm_response)
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable
        return llm_response

    async def simple_response(
        self,
        text: str,
        processors: Optional[List[BaseProcessor]] = None,
        participant: Participant = None,
    ) -> LLMResponse[Response]:
        instructions = None
        if self.conversation is not None:
            instructions = self.conversation.instructions

        return await self.create_response(
            input=text,
            instructions=instructions,
        )

    @staticmethod
    def _normalize_message(openai_input: Any) -> List["Message"]:
        from stream_agents.core.agents.conversation import Message

        # standardize on input
        if isinstance(openai_input, str):
            openai_input = [dict(content=openai_input, role="user", type="message")]
        elif not isinstance(openai_input, List):
            openai_input = [openai_input]

        messages: List[Message] = []
        for i in openai_input:
            message = Message(original=i, content=i["content"])
            messages.append(message)

        return messages
