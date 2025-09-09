import datetime
from typing import Optional, List, Any

from getstream.chat.client import ChatClient
from getstream.models import MessageRequest, ChannelResponse, MessageResponse
from dataclasses import dataclass


@dataclass
class Message:
    original: Any  # the original openai, claude or gemini message
    content: str
    timestamp: Optional[datetime.datetime] = None
    role: Optional[str] = None
    user_id: Optional[str] = None

    def __post_init__(self):
        self.timestamp = datetime.datetime.now()


class Conversation:
    def __init__(
        self,
        instructions: str,
        messages: List[Message],
    ):
        self.instructions = instructions
        self.messages = messages


class InMemoryConversation(Conversation):
    messages: List[Message]
    last_message: Optional[MessageResponse]

    def __init__(self, instructions: str, messages: List[Message]):
        super().__init__(instructions, messages)
        self.last_message = None

    def add_messages(self, messages: List[Message]):
        for m in messages:
            self.messages.append(m)

    def add_message(self, message: Message):
        self.messages.append(message)

    def finish_last_message(self, text: str):
        """Mark the last message as finished (not generating)."""
        if self.last_message:
            self.last_message.text = text
            self.last_message = None

    def update_last_message(self, input_text: str, user_id: str = "test"):
        """Update or create a partial message (for streaming responses)."""
        # When the user is speaking, or when the agent is generating
        if self.last_message is None:
            message = MessageRequest(text=input_text, user_id=user_id, type="regular")
            self.last_message = message
        else:
            self.last_message.text += input_text

    def partial_update_message(self, text: str, user):
        """Handle partial transcript updates."""
        user_id = user.user_id if user and hasattr(user, "user_id") else "unknown"
        self.update_last_message(text, user_id)


class StreamConversation(InMemoryConversation):
    """
    Persists the message history to a stream channel & messages
    """

    channel: ChannelResponse
    chat_client: ChatClient

    def __init__(
        self,
        instructions: str,
        messages: List[Message],
        channel: ChannelResponse,
        chat_client: ChatClient,
    ):
        super().__init__(instructions, messages)
        self.channel = channel
        self.chat_client = chat_client

    def _send_message(self, message):
        response = self.chat_client.send_message(
            self.channel.type, self.channel.id, message
        )
        return response

    def add_text_message(self, input_text: str, user_id: str) -> None:
        """Add a message to the conversation (text/user form)."""
        message = MessageRequest(text=input_text, user_id=user_id)
        self.messages.append(message)  # type: ignore[arg-type]
        self._send_message(message)

    def finish_last_message(self, text: str):
        """Mark the last message as finished (not generating)."""
        if self.last_message:
            self.last_message.text = text
            self.chat_client.update_message_partial(
                self.last_message.id,
                user_id=self.last_message.user.id,
                set={"text": text, "generating": False},
            )
            self.last_message = None

    def update_last_message(self, input_text: str, user_id: str = "test"):
        """Update or create a partial message (for streaming responses)."""
        # When the user is speaking, or when the agent is generating
        if self.last_message is None:
            message = MessageRequest(text=input_text, user_id=user_id, type="regular")
            response = self._send_message(message)
            self.last_message = response.data.message
        else:
            self.last_message.text += input_text
            self.chat_client.update_message_partial(
                self.last_message.id,
                user_id=self.last_message.user.id,
                set={"text": self.last_message.text, "generating": True},
            )

    def partial_update_message(self, text: str, user):
        """Handle partial transcript updates."""
        user_id = user.user_id if user and hasattr(user, "user_id") else "unknown"
        self.update_last_message(text, user_id)
