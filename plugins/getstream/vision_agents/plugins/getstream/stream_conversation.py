import logging
import threading
import queue
import time
from typing import List, Dict

from getstream.chat.client import ChatClient
from getstream.models import MessageRequest, ChannelResponse

from vision_agents.core.agents.conversation import InMemoryConversation, Message


logger = logging.getLogger(__name__)


class StreamConversation(InMemoryConversation):
    """
    Persists the message history to a stream channel & messages
    """
    messages: List[Message]

    # maps internal ids to stream message ids
    internal_ids_to_stream_ids: Dict[str, str]

    channel: ChannelResponse
    chat_client: ChatClient

    def __init__(self, instructions: str, messages: List[Message], channel: ChannelResponse, chat_client: ChatClient):
        super().__init__(instructions, messages)
        self.messages = messages
        self.channel = channel
        self.chat_client = chat_client
        self.internal_ids_to_stream_ids = {}

        # Initialize the worker thread for API calls
        self._api_queue: queue.Queue = queue.Queue()
        self._shutdown = False
        self._worker_thread = threading.Thread(target=self._api_worker, daemon=True, name="StreamConversation-APIWorker")
        self._worker_thread.start()
        self._pending_operations = 0
        self._operations_lock = threading.Lock()
        logger.info(f"Started API worker thread for channel {channel.id}")

    def _api_worker(self):
        """Worker thread that processes Stream API calls."""
        logger.debug("API worker thread started")
        while not self._shutdown:
            try:
                # Get operation from queue with timeout to check shutdown periodically
                operation = self._api_queue.get(timeout=0.1)

                try:
                    op_type = operation["type"]
                    logger.debug(f"Processing API operation: {op_type}")

                    if op_type == "send_message":
                        response = self.chat_client.send_message(
                            operation["channel_type"],
                            operation["channel_id"],
                            operation["request"]
                        )
                        # Store the mapping
                        self.internal_ids_to_stream_ids[operation["internal_id"]] = response.data.message.id
                        operation["stream_id"] = response.data.message.id

                    elif op_type == "update_message_partial":
                        self.chat_client.update_message_partial(
                            operation["stream_id"],
                            user_id=operation["user_id"],
                            set=operation["set_data"]
                        )

                    elif op_type == "ephemeral_message_update":
                        self.chat_client.ephemeral_message_update(
                            operation["stream_id"],
                            user_id=operation["user_id"],
                            set=operation["set_data"]
                        )

                    logger.debug(f"Successfully processed API operation: {op_type}")

                except Exception as e:
                    logger.error(f"Error processing API operation {operation.get('type', 'unknown')}: {e}")
                    # Continue processing other operations even if one fails

                finally:
                    # Decrement pending operations counter
                    with self._operations_lock:
                        self._pending_operations -= 1

            except queue.Empty:
                # Timeout reached, loop back to check shutdown flag
                continue
            except Exception as e:
                logger.error(f"Unexpected error in API worker thread: {e}")
                time.sleep(0.1)  # Brief pause before continuing

        logger.debug("API worker thread shutting down")

    def wait_for_pending_operations(self, timeout: float = 5.0) -> bool:
        """Wait for all pending API operations to complete.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            True if all operations completed, False if timeout reached.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self._operations_lock:
                if self._pending_operations == 0:
                    return True
            time.sleep(0.01)  # Small sleep to avoid busy waiting

        with self._operations_lock:
            remaining = self._pending_operations
        if remaining > 0:
            logger.warning(f"Timeout waiting for {remaining} pending operations")
        return False

    def shutdown(self):
        """Shutdown the worker thread gracefully."""
        logger.info("Shutting down API worker thread")
        self._shutdown = True
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
            if self._worker_thread.is_alive():
                logger.warning("API worker thread did not shut down cleanly")

    def add_message(self, message: Message, completed: bool = True):
        """Add a message to the Stream conversation.

        Args:
            message: The Message object to add
            completed: If True, mark the message as completed using update_message_partial.
                      If False, mark as still generating using ephemeral_message_update.

        Returns:
            None (operations are processed asynchronously)
        """
        self.messages.append(message)

        # Queue the send_message operation
        request = MessageRequest(text=message.content, user_id=message.user_id)
        send_op = {
            "type": "send_message",
            "channel_type": self.channel.type,
            "channel_id": self.channel.id,
            "request": request,
            "internal_id": message.id,
        }

        # Increment pending operations counter
        with self._operations_lock:
            self._pending_operations += 1

        self._api_queue.put(send_op)

        # Queue the update operation (will use the stream_id once send_message completes)
        # We need to wait for the send operation to complete first
        # So we'll handle this in a second operation that waits for the stream_id
        def queue_update_operation():
            # Wait for the stream_id to be available
            max_wait = 5.0
            start_time = time.time()
            while time.time() - start_time < max_wait:
                stream_id = self.internal_ids_to_stream_ids.get(message.id if message.id else "")
                if stream_id:
                    update_op = {
                        "type": "update_message_partial" if completed else "ephemeral_message_update",
                        "stream_id": stream_id,
                        "user_id": message.user_id,
                        "set_data": {"text": message.content, "generating": not completed},
                    }
                    with self._operations_lock:
                        self._pending_operations += 1
                    self._api_queue.put(update_op)
                    return
                time.sleep(0.01)
            logger.error(f"Timeout waiting for stream_id for message {message.id}")

        # Queue the update in a separate thread to avoid blocking
        threading.Thread(target=queue_update_operation, daemon=True).start()

    def update_message(self, message_id: str, input_text: str, user_id: str, replace_content: bool, completed: bool):
        """Update a message in the Stream conversation.

        This method updates both the local message content and queues the Stream API sync.
        If the message doesn't exist, it creates a new one.

        Args:
            message_id: The ID of the message to update
            input_text: The text content to set or append
            user_id: The ID of the user who owns the message
            replace_content: If True, replace the entire message content. If False, append to existing content.
            completed: If True, mark the message as completed using update_message_partial.
                      If False, mark as still generating using ephemeral_message_update.

        Returns:
            None (operations are processed asynchronously)
        """
        # First, update the local message using the superclass logic
        super().update_message(message_id, input_text, user_id, replace_content, completed)

        # Get the updated message for Stream API sync
        message = self.lookup(message_id)
        if message is None:
            # This shouldn't happen after super().update_message, but handle gracefully
            logger.warning(f"message {message_id} not found after update")
            return None

        stream_id = self.internal_ids_to_stream_ids.get(message_id)
        if stream_id is None:
            logger.warning(f"stream_id for message {message_id} not found, skipping Stream API update")
            return None

        # Queue the update operation
        update_op = {
            "type": "update_message_partial" if completed else "ephemeral_message_update",
            "stream_id": stream_id,
            "user_id": message.user_id,
            "set_data": {"text": message.content, "generating": not completed},
        }

        with self._operations_lock:
            self._pending_operations += 1

        return self._api_queue.put(update_op)

    def __del__(self):
        """Cleanup when the conversation is destroyed."""
        try:
            self.shutdown()
        except Exception as e:
            logger.error(f"Error during StreamConversation cleanup: {e}")