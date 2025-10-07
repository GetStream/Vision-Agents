import datetime
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Optional, List, Any

from dataclasses import dataclass


logger = logging.getLogger(__name__)

@dataclass
class Message:
    """A single utterance or assistant message within a conversation.

    Attributes:
        content: Text content of the message.
        original: Optional provider-native object for this message.
        timestamp: Time the message was created (auto-filled on init).
        role: Role of the sender (e.g., "user", "assistant").
        user_id: Logical user identifier associated with the message.
        id: Unique message identifier (auto-generated if not provided).
    """
    content: str
    original: Optional[Any] = None  # the original openai, claude or gemini message
    timestamp: Optional[datetime.datetime] = None
    role: Optional[str] = None
    user_id: Optional[str] = None
    id: Optional[str] = None

    def __post_init__(self):
        self.id = self.id or str(uuid.uuid4())
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
        """Create a conversation container.

        Args:
            instructions: System instructions that guide the assistant.
            messages: Initial message history.
        """
        self.instructions = instructions
        self.messages = [m for m in messages]

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
        # Message IDs and user_ids are always set by add_message
        assert added_message.id is not None, "Message ID should be set by add_message"
        assert added_message.user_id is not None, "User ID should be set by add_message"
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
        """Create an in-memory conversation holder.

        Stores messages in a local list and performs updates in place. Useful for
        tests and local development, or as a base for provider-backed
        conversations.
        """
        super().__init__(instructions, messages)

    def lookup(self, id: str) -> Optional[Message]:
        """Find a message by ID. Needed by StreamConversation

        Args:
            id: Message identifier to lookup.

        Returns:
            The `Message` if found, otherwise None.
        """
        msgs = [m for m in self.messages if m.id == id]
        if msgs:
            return msgs[0]
        return None

    def add_message(self, message: Message, completed: bool = True):
        """Append a message to the in-memory list and return None.

        The `completed` flag is not used for in-memory conversations.
        """
        self.messages.append(message)
        # In-memory conversation doesn't need to handle completed flag
        return None

    def update_message(self, message_id: str, input_text: str, user_id: str, replace_content: bool, completed: bool):
        """Update or create a message in-memory.

        If the message is not found, a new one is created with the given id.

        Args:
            message_id: Target message identifier.
            input_text: Text to set (replace) or append.
            user_id: Owner user id for the message.
            replace_content: If True, replace content; otherwise append.
            completed: Ignored for in-memory conversations.
        """
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

