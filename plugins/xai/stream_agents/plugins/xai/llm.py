from typing import Optional, List, Any, TYPE_CHECKING
from xai_sdk import AsyncClient
from xai_sdk.chat import system, user, Response, Chunk
from xai_sdk.proto import chat_pb2

from stream_agents.core.llm.llm import LLM, LLMResponse
from stream_agents.core.llm.types import StandardizedTextDeltaEvent
from stream_agents.core.processors import BaseProcessor

if TYPE_CHECKING:
    from stream_agents.core.agents.conversation import Message
    from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant
    from xai_sdk.aio.chat import Chat
else:
    from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant


class XAILLM(LLM):
    """
    The XAILLM class provides full/native access to the xAI SDK methods.
    It only standardizes the minimal feature set that's needed for the agent integration.

    The agent requires that we standardize:
    - sharing instructions
    - keeping conversation history
    - response normalization

    Notes on the xAI integration
    - the native method is called create_response (maps to xAI chat.sample())
    - history is maintained using the chat object's append method

    Examples:

        from stream_agents.plugins import xai
        llm = xai.LLM(model="grok-beta")

    """

    def __init__(
        self,
        model: str = "grok-4",
        api_key: Optional[str] = None,
        client: Optional[AsyncClient] = None,
    ):
        """
        Initialize the XAILLM class.

        Args:
            model (str): The xAI model to use. Defaults to "grok-4"
            api_key: optional API key. by default loads from XAI_API_KEY
            client: optional xAI client. by default creates a new client object.
        """
        super().__init__()
        self.model = model
        self.xai_chat: Optional["Chat"] = None
        self.conversation = None

        if client is not None:
            self.client = client
        elif api_key is not None and api_key != "":
            self.client = AsyncClient(api_key=api_key)
        else:
            self.client = AsyncClient()

    async def simple_response(
        self,
        text: str,
        processors: Optional[List[BaseProcessor]] = None,
        participant: Optional[Participant] = None,
    ):
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

    async def create_response(self, *args: Any, **kwargs: Any) -> LLMResponse[Response]:
        """
        create_response gives you full support/access to the native xAI chat.sample() and chat.stream() methods
        this method wraps the xAI method and ensures we broadcast an event which the agent class hooks into
        """
        input_text = kwargs.get("input", "")
        instructions = kwargs.get("instructions", "")
        model = kwargs.get("model", self.model)
        stream = kwargs.get("stream", True)

        # Create or reuse chat session
        if not self.xai_chat:
            messages = []
            if instructions:
                messages.append(system(instructions))
            self.xai_chat = self.client.chat.create(model=model, messages=messages)

        # Add user message
        assert self.xai_chat is not None
        self.xai_chat.append(user(input_text))

        self.emit("before_llm_response", self._normalize_message(input_text))

        # Get response based on streaming preference
        if stream:
            # Handle streaming response
            llm_response: Optional[LLMResponse[Response]] = None
            assert self.xai_chat is not None
            async for response, chunk in self.xai_chat.stream():
                llm_response_optional = self._standardize_and_emit_chunk(
                    chunk, response
                )
                if llm_response_optional is not None:
                    llm_response = llm_response_optional

            # Add response to chat history
            if llm_response and llm_response.original:
                assert self.xai_chat is not None
                self.xai_chat.append(llm_response.original)
        else:
            # Handle non-streaming response
            assert self.xai_chat is not None
            response = await self.xai_chat.sample()
            llm_response = LLMResponse[Response](response, response.content)

            # Add response to chat history
            assert self.xai_chat is not None
            self.xai_chat.append(response)

        self.emit("after_llm_response", llm_response)

        return llm_response or LLMResponse[Response](
            Response(chat_pb2.GetChatCompletionResponse(), 0), ""
        )

    @staticmethod
    def _normalize_message(input_text: str) -> List["Message"]:
        """
        Takes the input text and standardizes it so we can store it in chat
        """
        from stream_agents.core.agents.conversation import Message

        # Create a standardized message from input text
        message = Message(
            original={"content": input_text, "role": "user", "type": "message"},
            content=input_text,
        )

        return [message]

    def _standardize_and_emit_chunk(
        self, chunk: Chunk, response: Response
    ) -> Optional[LLMResponse[Response]]:
        """
        Forwards the chunk events and also send out a standardized version (the agent class hooks into that)
        """
        # Emit the raw chunk event
        self.emit("chunk", chunk)

        # Emit standardized delta events for content
        if chunk.content:
            standardized_event = StandardizedTextDeltaEvent(
                content_index=0,  # xAI doesn't have content_index
                item_id=chunk.proto.id if hasattr(chunk.proto, "id") else "",
                output_index=0,  # xAI doesn't have output_index
                sequence_number=0,  # xAI doesn't have sequence_number
                type="response.output_text.delta",
                delta=chunk.content,
            )
            self.emit("standardized.output_text.delta", standardized_event)

        # Check if this is the final chunk (finish_reason indicates completion)
        if chunk.choices and chunk.choices[0].finish_reason:
            # This is the final chunk, return the complete response
            llm_response = LLMResponse[Response](response, response.content)
            self.emit("standardized.response.completed", llm_response)
            return llm_response

        return None
