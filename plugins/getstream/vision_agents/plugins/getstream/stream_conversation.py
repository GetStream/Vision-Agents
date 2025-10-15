import logging
from typing import List, Dict

from getstream.models import MessageRequest
from getstream.chat.async_channel import Channel
from getstream.base import StreamAPIException

from vision_agents.core.agents.conversation import (
    Conversation,
    Message,
    MessageState,
)

logger = logging.getLogger(__name__)


class StreamConversation(Conversation):
    """Persists the message history to a Stream channel & messages."""
    
    # Maps internal message IDs to Stream message IDs
    internal_ids_to_stream_ids: Dict[str, str]
    channel: Channel

    def __init__(self, instructions: str, messages: List[Message], channel: Channel):
        super().__init__(instructions, messages)
        self.channel = channel
        self.internal_ids_to_stream_ids = {}

    async def _sync_to_backend(self, message: Message, state: MessageState, completed: bool):
        """Sync message to Stream Chat API.
        
        Args:
            message: The message to sync
            state: The message's internal state
            completed: If True, finalize the message. If False, mark as still generating.
        """
        if not state.created_in_backend:
            # Create message in Stream for the first time
            try:
                request = MessageRequest(
                    text=message.content,
                    user_id=message.user_id,
                    custom={"generating": not completed}
                )
                response = await self.channel.send_message(request)
                
                if response.data.message.type == "error":
                    raise StreamAPIException(response=response.__response)
                
                # Store mapping
                stream_message_id = response.data.message.id
                self.internal_ids_to_stream_ids[message.id] = stream_message_id
                state.created_in_backend = True
                
                # Store stream ID in state for subsequent updates
                state.stream_message_id = stream_message_id
                
                logger.debug(
                    f"Created Stream message {stream_message_id} for internal message {message.id}"
                )
            except Exception as e:
                logger.error(f"Failed to create Stream message for {message.id}: {e}")
                raise
        else:
            # Update existing message in Stream
            stream_message_id = getattr(state, 'stream_message_id', None)
            if stream_message_id is None:
                # Fallback to lookup
                stream_message_id = self.internal_ids_to_stream_ids.get(message.id)
            
            if stream_message_id is None:
                logger.warning(
                    f"stream_id for message {message.id} not found, skipping Stream API update"
                )
                return
            
            try:
                if completed:
                    # Finalize with update_message_partial
                    await self.channel.client.update_message_partial(
                        stream_message_id,
                        user_id=message.user_id,
                        set={"text": message.content, "generating": False}
                    )
                    logger.debug(f"Finalized Stream message {stream_message_id}")
                else:
                    # Update with ephemeral_message_update (still generating)
                    await self.channel.client.ephemeral_message_update(
                        stream_message_id,
                        user_id=message.user_id,
                        set={"text": message.content, "generating": True}
                    )
                    logger.debug(
                        f"Updated Stream message {stream_message_id} (generating)"
                    )
            except Exception as e:
                logger.error(f"Failed to update Stream message {stream_message_id}: {e}")
                # Don't re-raise on update failures - message is in memory
