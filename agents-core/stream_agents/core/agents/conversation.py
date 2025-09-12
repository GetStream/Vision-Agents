import datetime
import logging
import uuid
import threading
import queue
import time
from abc import ABC, abstractmethod
from typing import Optional, List, Any, Dict

from getstream.chat.client import ChatClient
from getstream.models import MessageRequest, ChannelResponse
from dataclasses import dataclass


logger = logging.getLogger(__name__)

@dataclass
class Message:
    original: Any  # the original openai, claude or gemini message
    content: str
    timestamp: Optional[datetime.datetime] = None
    role: Optional[str] = None
    user_id: Optional[str] = None
    id: Optional[str] = None

    def __post_init__(self):
        self.timestamp = datetime.datetime.now()


@dataclass
class StreamHandle:
    """Handle for managing a streaming message.
    
    This lightweight object is returned when starting a streaming message
    and must be passed to subsequent update operations. It encapsulates
    the message ID and user ID, preventing accidental cross-contamination
    between concurrent streams.
    
    Example:
        # Start a streaming message
        handle = conversation.start_streaming_message(role="assistant")
        
        # Update the message using the handle
        conversation.append_to_message(handle, "Hello")
        conversation.append_to_message(handle, " world!")
        
        # Complete the message
        conversation.complete_message(handle)
    """
    message_id: str
    user_id: str


class Conversation(ABC):
    def __init__(
        self,
        instructions: str,
        messages: List[Message],
    ):
        self.instructions = instructions
        self.messages = []

        for m in messages:
            if not hasattr(m, 'id') or m.id is None:
                m.id = uuid.uuid4().__str__()
            self.messages.append(m)
    
    @abstractmethod
    def add_message(self, message: Message, completed: bool = True):
        """Add a message to the conversation.
        
        Args:
            message: The Message object to add
            completed: If True, mark the message as completed (not generating). 
                      If False, mark as still generating. Defaults to True.
        
        Returns:
            The result of the add operation (implementation-specific)
        """
        ...
    
    @abstractmethod
    def update_message(self, message_id: str, input_text: str, user_id: str, replace_content: bool, completed: bool):
        """Update an existing message or create a new one if not found.
        
        Args:
            message_id: The ID of the message to update
            input_text: The text content to set or append
            user_id: The ID of the user who owns the message
            replace_content: If True, replace the entire message content. If False, append to existing content.
            completed: If True, mark the message as completed (not generating). If False, mark as still generating.
        
        Returns:
            The result of the update operation (implementation-specific)
        """
        ...
    
    # Streaming message convenience methods
    def start_streaming_message(self, role: str = "assistant", user_id: Optional[str] = None, 
                               initial_content: str = "") -> StreamHandle:
        """Start a new streaming message and return a handle for subsequent operations.
        
        This method simplifies the management of streaming messages by returning a handle
        that encapsulates the message ID and user ID. Use the handle with append_to_message,
        replace_message, and complete_message methods.
        
        Args:
            role: The role of the message sender (default: "assistant")
            user_id: The ID of the user (default: same as role)
            initial_content: Initial content for the message (default: empty string)
            
        Returns:
            StreamHandle: A handle to use for subsequent operations on this message
            
        Example:
            # Simple usage
            handle = conversation.start_streaming_message()
            conversation.append_to_message(handle, "Processing...")
            conversation.replace_message(handle, "Here's the answer: ")
            conversation.append_to_message(handle, "42")
            conversation.complete_message(handle)
            
            # Multiple concurrent streams
            user_handle = conversation.start_streaming_message(role="user", user_id="user123")
            assistant_handle = conversation.start_streaming_message(role="assistant")
            
            # Update both independently
            conversation.append_to_message(user_handle, "Hello")
            conversation.append_to_message(assistant_handle, "Hi there!")
            
            # Complete in any order
            conversation.complete_message(user_handle)
            conversation.complete_message(assistant_handle)
        """
        message = Message(
            original=None,
            content=initial_content,
            role=role,
            user_id=user_id or role,
            id=None  # Will be assigned during add
        )
        self.add_message(message, completed=False)
        # The message now has an ID assigned by the add_message flow
        # Find it in the messages list (it's the last one added)
        added_message = self.messages[-1]
        return StreamHandle(message_id=added_message.id, user_id=added_message.user_id)
    
    def append_to_message(self, handle: StreamHandle, text: str):
        """Append text to a streaming message identified by the handle.
        
        Args:
            handle: The StreamHandle returned by start_streaming_message
            text: Text to append to the message
        """
        self.update_message(
            message_id=handle.message_id,
            input_text=text,
            user_id=handle.user_id,
            replace_content=False,
            completed=False
        )
    
    def replace_message(self, handle: StreamHandle, text: str):
        """Replace the content of a streaming message identified by the handle.
        
        Args:
            handle: The StreamHandle returned by start_streaming_message
            text: Text to replace the message content with
        """
        self.update_message(
            message_id=handle.message_id,
            input_text=text,
            user_id=handle.user_id,
            replace_content=True,
            completed=False
        )
    
    def complete_message(self, handle: StreamHandle):
        """Mark a streaming message as completed.
        
        Args:
            handle: The StreamHandle returned by start_streaming_message
        """
        # We need to find the message to get its current content
        # so we can set completed without changing the content
        message = next((msg for msg in self.messages if msg.id == handle.message_id), None)
        if message:
            # Use replace mode with the current content to avoid space issues
            self.update_message(
                message_id=handle.message_id,
                input_text=message.content,
                user_id=handle.user_id,
                replace_content=True,
                completed=True
            )


class InMemoryConversation(Conversation):
    messages: List[Message]

    def __init__(self, instructions: str, messages: List[Message]):
        super().__init__(instructions, messages)

    def lookup(self, id: str) -> Optional[Message]:
        """Internal method to find message by ID - needed by StreamConversation"""
        msgs = [m for m in self.messages if m.id == id]
        if msgs:
            return msgs[0]
        return None

    def add_message(self, message: Message, completed: bool = True):
        # Ensure message has an ID
        if not hasattr(message, 'id') or message.id is None:
            message.id = uuid.uuid4().__str__()
        self.messages.append(message)
        # In-memory conversation doesn't need to handle completed flag
        return None

    def update_message(self, message_id: str, input_text: str, user_id: str, replace_content: bool, completed: bool):
        # Find the message by id
        message = self.lookup(message_id)
        
        if message is None:
            logger.info(f"message {message_id} not found, create one instead")
            return self.add_message(Message(user_id=user_id, id=message_id, content=input_text, original=None), completed=completed)

        if replace_content:
            message.content = input_text
        else:
            message.content += input_text

        # In-memory conversation just updates the message, no external API call
        return None


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
        # Ensure message has an ID
        if not hasattr(message, 'id') or message.id is None:
            message.id = uuid.uuid4().__str__()
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
                stream_id = self.internal_ids_to_stream_ids.get(message.id)
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
        
        self._api_queue.put(update_op)

    # Backward compatibility methods for easier migration
    def add_text_message(self, input_text: str, user_id: str) -> None:
        """Add a message to the conversation (text/user form) - backward compatibility method."""
        message = Message(
            original=None,
            content=input_text,
            role="user",
            user_id=user_id
        )
        self.add_message(message, completed=True)
    
    def finish_last_message(self, text: str):
        """Backward compatibility wrapper - mark the last message as completed with final text."""
        if self.messages:
            last_msg = self.messages[-1]
            self.update_message(last_msg.id, text, last_msg.user_id, replace_content=True, completed=True)
    
    def partial_update_message(self, text: str, user_metadata: Optional[Any] = None):
        """Backward compatibility wrapper - update the last message partially (append text)."""
        if self.messages:
            last_msg = self.messages[-1]
            # If user_metadata is provided and has user_id, use it; otherwise use the message's user_id
            user_id = getattr(user_metadata, 'user_id', last_msg.user_id) if user_metadata else last_msg.user_id
            self.update_message(last_msg.id, text, user_id, replace_content=False, completed=False)
        else:
            # No messages yet, create a new one
            user_id = getattr(user_metadata, 'user_id', 'unknown') if user_metadata else 'unknown'
            message = Message(
                original=None,
                content=text,
                role="assistant",
                user_id=user_id
            )
            self.add_message(message, completed=False)
    
    def __del__(self):
        """Cleanup when the conversation is destroyed."""
        try:
            self.shutdown()
        except Exception as e:
            logger.error(f"Error during StreamConversation cleanup: {e}")