import datetime
from dataclasses import dataclass
from typing import Optional, List, ParamSpec, TypeVar, Callable, TYPE_CHECKING, Literal

from openai import OpenAI, Stream
from openai.lib.streaming.responses import ResponseStreamEvent
from openai.resources.responses import Responses

from getstream.models import Response
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant
from openai.types.responses import ResponseCompletedEvent, ResponseOutputText, ResponseTextDeltaEvent

from stream_agents.core.llm.llm import LLM, LLMResponse

from stream_agents.core.processors import BaseProcessor

if TYPE_CHECKING:
    from stream_agents.core.agents.conversation import Message


P = ParamSpec("P")
R = TypeVar("R")

def use_create(fn: Callable[P, R]) -> Callable[P, R]:
    return fn

# TODO: somehow this isn't right, docs aren't great: https://peps.python.org/pep-0612/
bound = use_create(Responses.create)

@dataclass
class StandardizedTextDeltaEvent:
    content_index: int
    """The index of the content part that the text delta was added to."""

    delta: str
    """The text delta that was added."""

    item_id: str
    """The ID of the output item that the text delta was added to."""

    output_index: int
    """The index of the output item that the text delta was added to."""

    sequence_number: int
    """The sequence number for this event."""

    type: Literal["response.output_text.delta"]
    """The type of the event. Always `response.output_text.delta`."""

class OpenAILLM(LLM):
    '''
    The goal is to expose the regular/native openAI SDK methods,
    and only standardize the minimal feature set that's needed for the agent integration.

    The agent requires that we standardize:
    - sharing instructions
    - keeping conversation history
    - response normalization

    Notes on the OpenAI integration
    - the native method is called create_response (maps 1-1 to responses.create)
    - history is maintained using conversation.create()

    TODO:
    - proper typing for args, kwargs
    '''
    def __init__(self, model: str, api_key: Optional[str] = None, client: Optional[OpenAI] = None):
        super().__init__()
        self.model = model
        self.openai_conversation = None
        self.conversation = None


        if client is not None:
            self.client = client
        elif api_key is not None and api_key is not "":
            self.openai_conversation = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI()

    async def create_response(self, *args: P.args, **kwargs: P.kwargs) -> LLMResponse:
        if "model" not in kwargs:
            kwargs["model"] = self.model
        if "stream" not in kwargs:
            kwargs["stream"] = True

        if not self.openai_conversation:
            self.openai_conversation = self.client.conversations.create()
        kwargs["conversation"] = self.openai_conversation.id

        if hasattr(self, "before_response_listener"):
            self.before_response_listener(self._normalize_message(kwargs["input"]))
        response = self.client.responses.create(
            *args, **kwargs
        )

        if isinstance(response, Response):
            llm_response = LLMResponse[Response](response, response.output_text)
        elif isinstance(response, Stream):
            response : Stream[ResponseStreamEvent] = response
            # handle both streaming and non-streaming response types
            for event in response:
                from pprint import pprint
                pprint(event.type)
                if event.type == "response.output_text.delta":
                    event : ResponseTextDeltaEvent = event

                    standardized_event = StandardizedTextDeltaEvent(
                        content_index=event.content_index,
                        item_id=event.item_id,
                        output_index=event.output_index,
                        sequence_number=event.sequence_number,
                        type=event.type,
                        delta=event.delta,
                    )
                    self.emit("standardized.output_text.delta", standardized_event)

                    pprint(event.delta)
                    print("\n")
                elif event.type == "response.completed":
                    event : ResponseCompletedEvent = event
                    llm_response = LLMResponse[Response](event.response, event.response.output_text)
                    self.emit("standardized.response.completed", llm_response)
                else:
                    pprint( event)

                self.emit(event.type, event)
                # standardize the events. emit both the native and standardized events?

                pass

        if hasattr(self, "after_response_listener"):
            await self.after_response_listener(llm_response)

        return llm_response

    async def simple_response(self, text: str, processors: Optional[List[BaseProcessor]] = None, participant: Participant = None):
        instructions = None
        if self.conversation is not None:
            instructions = self.conversation.instructions

        return await self.create_response(
            input=text,
            instructions=instructions,
        )

    @staticmethod
    def _normalize_message(openai_input) -> List["Message"]:
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
            message = Message(original=i, content = i["content"])
            messages.append(message)

        return messages