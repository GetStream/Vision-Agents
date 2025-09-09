from typing import Optional, List, TYPE_CHECKING, Any

from google import genai
from google.genai.types import GenerateContentResponse

from stream_agents.core.llm.llm import LLM, LLMResponse
from stream_agents.core.llm.types import StandardizedTextDeltaEvent

from stream_agents.core.processors import BaseProcessor

if TYPE_CHECKING:
    from stream_agents.core.agents.conversation import Message


class GeminiLLM(LLM):
    """
      The GeminiLLM class provides full/native access to the gemini SDK methods.
      It only standardized the minimal feature set that's needed for the agent integration.

      The agent requires that we standardize:
      - sharing instructions
      - keeping conversation history
      - response normalization

      Notes on the Gemini integration:
      - the native method is called send_message (maps 1-1 to chat.send_message_stream)
      - history is maintained in the gemini sdk (with the usage of client.chats.create(model=self.model))

      Examples:

          from stream_agents.plugins import gemini
          llm = gemini.LLM()
      """
    def __init__(self, model: str, api_key: Optional[str] = None, client: Optional[genai.Client] = None):
        """
        Initialize the GeminiLLM class.

        Args:
            model (str): The model to use.
            api_key: optional API key. by default loads from GOOGLE_API_KEY
            client: optional Anthropic client. by default creates a new client object.
        """
        super().__init__()
        self.model = model
        self.chat: Optional[Any] = None

        if client is not None:
            self.client = client
        else:
            self.client = genai.Client(api_key=api_key)

    async def simple_response(self, text: str, processors: Optional[List[BaseProcessor]] = None, participant: Optional[Any] = None) -> LLMResponse[Any]:
        """
        simple_response is a standardized way (across openai, claude, gemini etc.) to create a response.

        Args:
            text: The text to respond to
            processors: list of processors (which contain state) about the video/voice AI

        Examples:

            llm.simple_response("say hi to the user, be mean")
        """
        return await self.send_message(
            message=text
        )

    async def send_message(self, *args, **kwargs):
        """
        send_message gives you full support/access to the native Gemini chat send message method
        under the hood it calls chat.send_message_stream(*args, **kwargs)
        this method wraps and ensures we broadcast an event which the agent class hooks into
        """
        #if "model" not in kwargs:
        #    kwargs["model"] = self.model

        # initialize chat if needed
        if self.chat is None:
            self.chat = self.client.chats.create(model=self.model)

        self.emit("before_llm_response", self._normalize_message(kwargs["input"]))

        # Generate content using the client
        iterator = self.chat.send_message_stream(*args, **kwargs)
        for chunk in iterator:
            response_chunk: GenerateContentResponse = chunk
            llm_response_optional = self._standardize_and_emit_event(response_chunk)
            if llm_response_optional is not None:
                llm_response = llm_response_optional

        self.emit("after_llm_response", llm_response)

        # Return the LLM response
        return llm_response

    @staticmethod
    def _normalize_message(gemini_input) -> List["Message"]:
        from stream_agents.core.agents.conversation import Message
        
        # standardize on input
        if isinstance(gemini_input, str):
            gemini_input = [
                gemini_input
            ]

        if not isinstance(gemini_input, List):
            gemini_input = [gemini_input]

        messages = []
        for i in gemini_input:
            message = Message(original=i, content=i)
            messages.append(message)

        return messages

    def _standardize_and_emit_event(self, chunk: GenerateContentResponse):
        """
        Forwards the events and also send out a standardized version (the agent class hooks into that)
        """
        # forward the native event
        self.emit("gemini_response", chunk)
        
        # Check if response has text content
        if hasattr(chunk, 'text') and chunk.text:
            standardized_event = StandardizedTextDeltaEvent(
                content_index=0,
                item_id="",
                output_index=0,
                sequence_number=0,
                type="response.output_text.delta",
                delta=chunk.text,
            )
            self.emit("standardized.output_text.delta", standardized_event)
            
            # Return response for final text
            llm_response = LLMResponse(chunk, chunk.text)
            return llm_response
        return None