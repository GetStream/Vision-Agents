import asyncio
import datetime
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Optional, List, Any, Dict, TypeVar, Generic

from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Message:
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

    message_id: Optional[str] = None
    user_id: Optional[str] = None


class StreamingMessageHandler(ABC):
    """Base class for handling streaming messages with out-of-order content support."""

    def __init__(self, message_id: str, user_id: str, role: str):
        self.message_id = message_id
        self.user_id = user_id
        self.role = role
        self.content = ""  # Start with empty content
        self.is_finalized = False
        self.pending_fragments: Dict[int, str] = {}
        self.last_content_index = -1
        self._content_created = False
        self._apply_lock = asyncio.Lock()
        self._creation_lock = asyncio.Lock()  # Prevents race condition on _on_content_created

    async def append_content(
        self, text: str, content_index: Optional[int] = None, finalize: bool = False
    ):
        """Append content to the message, handling out-of-order delivery.

        Args:
            text: The text content to append
            content_index: Optional index for ordering (starts at 0). If None, uses next sequential index.
            finalize: If True, marks the message as completed
        """
        if self.is_finalized:
            raise RuntimeError(f"Cannot append to finalized message {self.message_id}")

        if content_index is None:
            content_index = self.last_content_index + 1

        # Store the fragment
        self.pending_fragments[content_index] = text

        # Try to apply pending fragments in order
        await self._apply_pending_fragments()

        if finalize:
            await self.finalize()

    async def set_content(self, text: str, finalize: bool = False):
        """Set the entire content of the message, replacing any existing content.

        Args:
            text: The complete text content
            finalize: If True, marks the message as completed
        """
        if self.is_finalized:
            raise RuntimeError(
                f"Cannot set content on finalized message {self.message_id}"
            )

        self.content = text
        self.pending_fragments.clear()
        self.last_content_index = -1

        # Use lock to prevent race condition: ensure only ONE call to _on_content_created
        async with self._creation_lock:
            if not self._content_created:
                await self._on_content_created()
                self._content_created = True
            else:
                await self._on_content_changed()

        if finalize:
            await self.finalize()

    async def finalize(self):
        """Mark the message as finalized and clean up."""
        if self.is_finalized:
            return

        self.is_finalized = True
        await self._on_finalized()

    async def update_message(self, text: str, replace_content: bool = True):
        """Update the message content and finalize it.
        
        Args:
            text: The text content to set/append
            replace_content: If True, replace the entire content. If False, append to existing content.
        """
        if replace_content:
            await self.set_content(text, finalize=True)
        else:
            await self.append_content(text, finalize=True)

    async def _apply_pending_fragments(self):
        """Apply pending fragments in sequential order."""
        async with self._apply_lock:
            fragments_applied = 0
            while True:
                next_index = self.last_content_index + 1
                if next_index in self.pending_fragments:
                    fragment = self.pending_fragments.pop(next_index)
                    self.content += fragment
                    self.last_content_index = next_index
                    fragments_applied += 1
                else:
                    break
            
            # Only send update to Stream after applying all available fragments
            if fragments_applied > 0:
                # Use lock to prevent race condition: ensure only ONE call to _on_content_created
                async with self._creation_lock:
                    if not self._content_created:
                        await self._on_content_created()
                        self._content_created = True
                    else:
                        await self._on_content_changed()

    @abstractmethod
    async def _on_content_created(self):
        """Called when the message content is first created. Implementation-specific logic goes here."""
        pass

    @abstractmethod
    async def _on_content_changed(self):
        """Called when the message content changes. Implementation-specific logic goes here."""
        pass

    @abstractmethod
    async def _on_finalized(self):
        """Called when the message is finalized. Implementation-specific cleanup goes here."""
        pass


# Type variable for the handler implementation
HandlerType = TypeVar("HandlerType", bound=StreamingMessageHandler)


class Conversation(ABC, Generic[HandlerType]):
    def __init__(
        self,
        instructions: str,
        messages: List[Message],
    ):
        self.instructions = instructions
        self.messages = [m for m in messages]
        self._streaming_handlers: Dict[str, HandlerType] = {}
        self._handlers_lock = asyncio.Lock()

    @abstractmethod
    async def add_message(self, message: Message, completed: bool = True):
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
    async def update_message(
        self,
        message_id: str,
        input_text: str,
        user_id: str,
        replace_content: bool,
        completed: bool,
    ):
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

    @abstractmethod
    def _create_handler(self, message_id: str, user_id: str, role: str) -> HandlerType:
        """Create a new handler instance. Implementation-specific."""
        pass

    async def get_streaming_message_handler(
        self, message_id: str
    ) -> Optional[HandlerType]:
        """Get an existing streaming message handler by ID.

        Args:
            message_id: The ID of the message handler to retrieve

        Returns:
            The handler if it exists, None otherwise
        """
        async with self._handlers_lock:
            return self._streaming_handlers.get(message_id)

    async def upsert_streaming_message_handler(
        self,
        message_id: str,
        user_id: str,
        role: str,
        content: str = "",
        content_index: Optional[int] = None,
    ) -> HandlerType:
        """Get or create a streaming message handler, ensuring only one exists per message_id.

        Args:
            message_id: The ID of the message handler
            user_id: The ID of the user
            role: The role of the message sender
            content: The content fragment to add (optional, defaults to empty string)
            content_index: The index of this content fragment (optional, auto-increments if not provided)

        Returns:
            The handler instance
        """
        async with self._handlers_lock:
            handler = self._streaming_handlers.get(message_id)

            if handler is None:
                # Check if a message with this ID already exists (completion event arrived first)
                existing_message = next(
                    (msg for msg in self.messages if msg.id == message_id), None
                )
                
                if existing_message is not None:
                    # Message already exists, don't create handler
                    return None
                
                # Create new handler
                handler = self._create_handler(message_id, user_id, role)
                self._streaming_handlers[message_id] = handler

        # Add content if provided (outside the lock to avoid blocking other operations)
        if content and handler is not None:
            await handler.append_content(content, content_index)

        return handler

    async def _remove_handler(self, message_id: str):
        """Remove a handler from the active handlers dict."""
        async with self._handlers_lock:
            self._streaming_handlers.pop(message_id, None)

    # Streaming message convenience methods
    async def start_streaming_message(
        self,
        role: str = "assistant",
        user_id: Optional[str] = None,
        initial_content: str = "",
    ) -> StreamHandle:
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
            id=None,  # Will be assigned during add
        )
        await self.add_message(message, completed=False)
        # The message now has an ID assigned by the add_message flow
        # Find it in the messages list (it's the last one added)
        added_message = self.messages[-1]
        return StreamHandle(message_id=added_message.id, user_id=added_message.user_id)

    async def append_to_message(self, handle: StreamHandle, text: str):
        """Append text to a streaming message identified by the handle.

        Args:
            handle: The StreamHandle returned by start_streaming_message
            text: Text to append to the message
        """
        await self.update_message(
            message_id=handle.message_id,
            input_text=text,
            user_id=handle.user_id,
            replace_content=False,
            completed=False,
        )

    async def replace_message(self, handle: StreamHandle, text: str):
        """Replace the content of a streaming message identified by the handle.

        Args:
            handle: The StreamHandle returned by start_streaming_message
            text: Text to replace the message content with
        """
        await self.update_message(
            message_id=handle.message_id,
            input_text=text,
            user_id=handle.user_id,
            replace_content=True,
            completed=False,
        )

    async def complete_message(self, handle: StreamHandle):
        """Mark a streaming message as completed.

        Args:
            handle: The StreamHandle returned by start_streaming_message
        """
        # We need to find the message to get its current content
        # so we can set completed without changing the content
        message = next(
            (msg for msg in self.messages if msg.id == handle.message_id), None
        )
        if message:
            # Use replace mode with the current content to avoid space issues
            await self.update_message(
                message_id=handle.message_id,
                input_text=message.content,
                user_id=handle.user_id,
                replace_content=True,
                completed=True,
            )


class InMemoryStreamingMessageHandler(StreamingMessageHandler):
    """In-memory implementation of StreamingMessageHandler."""

    def __init__(
        self,
        message_id: str,
        user_id: str,
        role: str,
        conversation: "InMemoryConversation",
    ):
        super().__init__(message_id, user_id, role)
        self.conversation = conversation

    async def _on_content_created(self):
        """Called when content is first created. For in-memory, same as content changed."""
        await self._on_content_changed()

    async def _on_content_changed(self):
        """Update the message in the conversation's messages list."""
        if self.conversation:
            # Find and update the message
            for message in self.conversation.messages:
                if message.id == self.message_id:
                    message.content = self.content
                    break

    async def _on_finalized(self):
        """Remove the handler from the conversation when finalized."""
        if self.conversation:
            await self.conversation._remove_handler(self.message_id)


class InMemoryConversation(Conversation[InMemoryStreamingMessageHandler]):
    messages: List[Message]

    def __init__(self, instructions: str, messages: List[Message]):
        super().__init__(instructions, messages)

    def lookup(self, id: str) -> Optional[Message]:
        """Internal method to find message by ID - needed by StreamConversation"""
        msgs = [m for m in self.messages if m.id == id]
        if msgs:
            return msgs[0]
        return None

    def _create_handler(
        self, message_id: str, user_id: str, role: str
    ) -> InMemoryStreamingMessageHandler:
        """Create a new in-memory streaming message handler."""
        # Create the message first (empty content initially, will be updated by handler)
        message = Message(content="", role=role, user_id=user_id, id=message_id)
        self.messages.append(message)

        # Create and return the handler
        return InMemoryStreamingMessageHandler(
            message_id=message_id, user_id=user_id, role=role, conversation=self
        )

    async def add_message(self, message: Message, completed: bool = True):
        self.messages.append(message)
        # In-memory conversation doesn't need to handle completed flag
        return None

    async def update_message(
        self,
        message_id: str,
        input_text: str,
        user_id: str,
        replace_content: bool,
        completed: bool,
    ):
        # Find the message by id
        message = self.lookup(message_id)

        if message is None:
            logger.info(f"message {message_id} not found, create one instead")
            return await self.add_message(
                Message(
                    user_id=user_id, id=message_id, content=input_text, original=None
                ),
                completed=completed,
            )

        if replace_content:
            message.content = input_text
        else:
            message.content += input_text

        # In-memory conversation just updates the message, no external API call
        return None
