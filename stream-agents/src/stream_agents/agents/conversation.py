from typing import Optional, List
from getstream.chat.client import ChatClient
from getstream.models import Message, MessageRequest, ChannelResponse, MessageResponse


class Conversation:
    """
    Manages conversation state and messaging for an agent.

    TODO:
    - stream-py support async
    - stream-py have a method to get a channel for a call
    - say text needs more data (type and author)
    """

    messages: List[Message]
    last_message: Optional[MessageResponse]
    channel: ChannelResponse
    chat_client: ChatClient

    def __init__(
        self,
        instructions: str,
        messages: List[Message],
        channel: ChannelResponse,
        chat_client: ChatClient,
    ):
        self.instructions = instructions
        self.messages = messages
        self.channel = channel
        self.chat_client = chat_client
        self.last_message = None

    def _send_message(self, message):
        response = self.chat_client.send_message(
            self.channel.type, self.channel.id, message
        )
        return response

    def add_message(self, input_text: str, user_id: str):
        """Add a message to the conversation."""
        message = MessageRequest(text=input_text, user_id=user_id)
        self.messages.append(message)
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
