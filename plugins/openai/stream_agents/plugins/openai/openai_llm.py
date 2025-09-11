from typing import Optional, List, ParamSpec, TypeVar, Callable, TYPE_CHECKING, Any

from openai import OpenAI, Stream
from openai.lib.streaming.responses import ResponseStreamEvent
from openai.resources.responses import Responses

from getstream.models import Response
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant
from openai.types.responses import ResponseCompletedEvent, ResponseTextDeltaEvent

from stream_agents.core.llm.llm import LLM, LLMResponse
from stream_agents.core.llm.types import StandardizedTextDeltaEvent

from stream_agents.core.processors import BaseProcessor

if TYPE_CHECKING:
    from stream_agents.core.agents.conversation import Message

P = ParamSpec("P")
R = TypeVar("R")


def use_create(fn: Callable[P, R]) -> Callable[P, R]:
    return fn


# TODO: somehow this isn't right, docs aren't great: https://peps.python.org/pep-0612/
bound = use_create(Responses.create)


class OpenAILLM(LLM):
    """
    The OpenAILLM class provides full/native access to the openAI SDK methods.
    It only standardized the minimal feature set that's needed for the agent integration.

    The agent requires that we standardize:
    - sharing instructions
    - keeping conversation history
    - response normalization

    Notes on the OpenAI integration
    - the native method is called create_response (maps 1-1 to responses.create)
    - history is maintained using conversation.create()

    Examples:

        from stream_agents.plugins import openai
        llm = openai.LLM(model="gpt-5")

    """

    def __init__(self, model: str, api_key: Optional[str] = None, client: Optional[OpenAI] = None):
        """
        Initialize the OpenAILLM class.

        Args:
            model (str): The OpenAI model to use. https://platform.openai.com/docs/models
            api_key: optional API key. by default loads from OPENAI_API_KEY
            client: optional OpenAI client. by default creates a new client object.
        """
        super().__init__()
        self.model = model
        self.openai_conversation: Optional[Any] = None
        self.conversation = None

        if client is not None:
            self.client = client
        elif api_key is not None and api_key != "":
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI()

    async def simple_response(self, text: str, processors: Optional[List[BaseProcessor]] = None,
                              participant: Participant = None):
        """
        simple_response is a standardized way (across openai, claude, gemini etc.) to create a response.

        Args:
            text: The text to respond to
            processors: list of processors (which contain state) about the video/voice AI
            participant: optionally the participant object

        Examples:

            llm.simple_response("say hi to the user, be mean")
        """
        instructions = None
        if self.conversation is not None:
            instructions = self.conversation.instructions

        return await self.create_response(
            input=text,
            instructions=instructions,
        )

    async def create_response(self, *args: P.args, **kwargs: P.kwargs) -> LLMResponse[Response]:
        """
        create_response gives you full support/access to the native openAI responses.create method
        this method wraps the openAI method and ensures we broadcast an event which the agent class hooks into
        """
        if "model" not in kwargs:
            kwargs["model"] = self.model
        if "stream" not in kwargs:
            kwargs["stream"] = True

        if not self.openai_conversation:
            self.openai_conversation = self.client.conversations.create()
        kwargs["conversation"] = self.openai_conversation.id

        self.emit("before_llm_response", self._normalize_message(kwargs["input"]))

        response = self.client.responses.create(
            *args, **kwargs
        )

        llm_response : Optional[LLMResponse[Response]] = None
        if isinstance(response, Response):
            llm_response = LLMResponse[Response](response, response.output_text)
        elif isinstance(response, Stream):
            stream_response: Stream[ResponseStreamEvent] = response
            # handle both streaming and non-streaming response types
            for event in stream_response:
                llm_response_optional = self._standardize_and_emit_event(event)
                if llm_response_optional is not None:
                    llm_response = llm_response_optional

        self.emit("after_llm_response", llm_response)

        return llm_response or LLMResponse[Response](Response(duration=0.0), "")

    @staticmethod
    def _normalize_message(openai_input) -> List["Message"]:
        """
        Takes the openAI list of messages and standardizes it so we can store it in chat
        """
        from stream_agents.core.agents.conversation import Message

        # standardize on input
        if isinstance(openai_input, str):
            openai_input = [
                dict(content=openai_input, role="user", type="message")
            ]
        elif not isinstance(openai_input, List):
            openai_input = [openai_input]

        messages: List[Message] = []
        for i in openai_input:
            message = Message(original=i, content=i["content"])
            messages.append(message)

        return messages

    def _standardize_and_emit_event(self, event: ResponseStreamEvent) -> Optional[LLMResponse]:
        """
        Forwards the events and also send out a standardized version (the agent class hooks into that)
        """
        # start by forwarding the native event
        self.emit(event.type, event)

        if event.type == "response.output_text.delta":
            # standardize the delta event
            delta_event: ResponseTextDeltaEvent = event
            standardized_event = StandardizedTextDeltaEvent(
                content_index=delta_event.content_index,
                item_id=delta_event.item_id,
                output_index=delta_event.output_index,
                sequence_number=delta_event.sequence_number,
                type=delta_event.type,
                delta=delta_event.delta,
            )
            self.emit("standardized.output_text.delta", standardized_event)
        elif event.type == "response.completed":
            # standardize the response event and return the llm response
            completed_event: ResponseCompletedEvent = event
            llm_response = LLMResponse[Response](completed_event.response, completed_event.response.output_text)
            self.emit("standardized.response.completed", llm_response)
            return llm_response
        return None