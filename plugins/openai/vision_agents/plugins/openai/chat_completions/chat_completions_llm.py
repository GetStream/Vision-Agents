import logging
from typing import Optional, cast

from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from vision_agents.core.llm.events import (
    LLMResponseChunkEvent,
    LLMResponseCompletedEvent,
)
from vision_agents.core.llm.llm import LLM, LLMResponseEvent
from vision_agents.core.processors import Processor

from .. import events

logger = logging.getLogger(__name__)


PLUGIN_NAME = "chat_completions_llm"


class ChatCompletionsLLM(LLM):
    """
    This plugin allows developers to easily interact with models that use Chat Completions API.
    The model is expected to accept text and respond with text.

    Features:
        - Streaming responses: Supports streaming text responses with real-time chunk events
        - Event-driven: Emits LLM events (chunks, completion, errors) for integration with other components

    Examples:

        from vision_agents.plugins import openai
        llm = openai.ChatCompletionsLLM(model="deepseek-ai/DeepSeek-V3.1")

    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Optional[AsyncOpenAI] = None,
    ):
        """
        Initialize the ChatCompletionsLLM class.

        Args:
            model (str): The model id to use.
            api_key: optional API key. By default, loads from OPENAI_API_KEY environment variable.
            base_url: optional base url. By default, loads from OPENAI_BASE_URL environment variable.
            client: optional `AsyncOpenAI` client. By default, creates a new client object.
        """
        super().__init__()
        self.model = model
        self.events.register_events_from_module(events)

        if client is not None:
            self._client = client
        else:
            self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def simple_response(
        self,
        text: str,
        processors: Optional[list[Processor]] = None,
        participant: Optional[Participant] = None,
    ) -> LLMResponseEvent:
        """
        simple_response is a standardized way to create an LLM response.

        This method is also called every time the new STT transcript is received.

        Args:
            text: The text to respond to.
            processors: list of processors (which contain state) about the video/voice AI.
            participant: the Participant object, optional. If not provided, the message will be sent with the "system" role.

        Examples:

            llm.simple_response("say hi to the user, be nice")
        """

        if self._conversation is None:
            # The agent hasn't joined the call yet.
            logger.warning(
                f'Cannot request a response from the LLM "{self.model}" - the conversation has not been initialized yet.'
            )
            return LLMResponseEvent(original=None, text="")

        # The simple_response is called directly without providing the participant -
        # assuming it's an initial prompt.
        if participant is None:
            await self._conversation.send_message(
                role="system", user_id="system", content=text
            )

        messages = await self._build_model_request()

        try:
            response = await self._client.chat.completions.create(  # type: ignore[arg-type]
                messages=messages,  # type: ignore[arg-type]
                model=self.model,
                stream=True,
            )
        except Exception as e:
            # Send an error event if the request failed
            logger.exception(f'Failed to get a response from the LLM "{self.model}"')
            self.events.send(
                events.LLMErrorEvent(
                    plugin_name=PLUGIN_NAME,
                    error_message=str(e),
                    event_data=e,
                )
            )
            return LLMResponseEvent(original=None, text="")

        i = 0
        llm_response_event: LLMResponseEvent[Optional[ChatCompletionChunk]] = (
            LLMResponseEvent(original=None, text="")
        )
        text_chunks: list[str] = []
        total_text = ""
        async for chunk in cast(AsyncStream[ChatCompletionChunk], response):
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            content = choice.delta.content
            finish_reason = choice.finish_reason

            if content:
                text_chunks.append(content)
                # Emit delta events for each response chunk.
                self.events.send(
                    LLMResponseChunkEvent(
                        plugin_name=PLUGIN_NAME,
                        content_index=None,
                        item_id=chunk.id,
                        output_index=0,
                        sequence_number=i,
                        delta=content,
                    )
                )

            if finish_reason:
                if finish_reason in ("length", "content"):
                    logger.warning(
                        f'The model finished the response due to reason "{finish_reason}"'
                    )
                # Emit the completion event when the response stream is finished.
                total_text = "".join(text_chunks)
                self.events.send(
                    LLMResponseCompletedEvent(
                        plugin_name=PLUGIN_NAME,
                        original=chunk,
                        text=total_text,
                        item_id=chunk.id,
                    )
                )

            llm_response_event = LLMResponseEvent(original=chunk, text=total_text)
            i += 1

        return llm_response_event

    async def _build_model_request(self) -> list[dict]:
        messages: list[dict] = []
        # Add Agent's instructions as system prompt.
        if self.instructions:
            messages.append({"role": "system", "content": self.instructions})

        # Add all messages from the conversation to the prompt
        if self._conversation is not None:
            for message in self._conversation.messages:
                messages.append({"role": message.role, "content": message.content})
        return messages
