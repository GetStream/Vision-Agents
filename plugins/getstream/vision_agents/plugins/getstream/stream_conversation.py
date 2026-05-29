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
    """Persists the message history to a Stream channel with automatic chunking."""

    # Maps internal message IDs to first Stream message ID (for backward compatibility)
    internal_ids_to_stream_ids: Dict[str, str]
    channel: Channel
    chunk_size: int

    def __init__(
        self,
        instructions: str,
        messages: List[Message],
        channel: Channel,
        chunk_size: int = 1000,
    ):
        """Initialize StreamConversation with automatic message chunking.

        Args:
            instructions: System instructions
            messages: Initial messages
            channel: Stream channel for persistence
            chunk_size: Maximum characters per message chunk (default 1000)
        """
        super().__init__(instructions, messages)
        self.channel = channel
        self.internal_ids_to_stream_ids = {}
        self.chunk_size = chunk_size

    async def _sync_to_backend(
        self, message: Message, state: MessageState, completed: bool
    ):
        """Sync message to Stream Chat API with automatic chunking.

        Args:
            message: The message to sync
            state: The message's internal state
            completed: If True, finalize the message. If False, mark as still generating.
        """
        # Split message into chunks (markdown-aware)
        chunks = self._smart_chunk(message.content, self.chunk_size)

        if not state.created_in_backend:
            # CREATE: Send all chunks to Stream
            await self._create_chunks(message, state, chunks, completed)
            state.created_in_backend = True
        else:
            # UPDATE: Update/create/delete chunks as needed
            await self._update_chunks(message, state, chunks, completed)

    async def _create_chunks(
        self, message: Message, state: MessageState, chunks: List[str], completed: bool
    ):
        """Create all chunks for a new message."""
        for i, chunk_text in enumerate(chunks):
            is_last_chunk = i == len(chunks) - 1

            request = MessageRequest(
                text=chunk_text,
                user_id=message.user_id,
                custom={
                    # Only the LAST chunk has "generating" flag
                    "generating": not completed if is_last_chunk else False,
                    "chunk_group": message.id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            )

            try:
                response = await self.channel.send_message(request)

                if response.data.message.type == "error":
                    raise StreamAPIException(response=response.__response)

                chunk_id = response.data.message.id
                state.backend_message_ids.append(chunk_id)

                logger.debug(
                    f"Created chunk {i + 1}/{len(chunks)} (Stream ID: {chunk_id}) "
                    f"for message {message.id}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to create chunk {i} for message {message.id}: {e}"
                )
                raise

        # Store first chunk ID for backward compatibility
        if state.backend_message_ids and message.id:
            self.internal_ids_to_stream_ids[message.id] = state.backend_message_ids[0]

    async def _update_chunks(
        self,
        message: Message,
        state: MessageState,
        new_chunks: List[str],
        completed: bool,
    ):
        """Update existing chunks, creating or deleting as needed."""
        old_chunk_count = len(state.backend_message_ids)
        new_chunk_count = len(new_chunks)

        # Update existing chunks
        for i in range(min(old_chunk_count, new_chunk_count)):
            chunk_id = state.backend_message_ids[i]
            chunk_text = new_chunks[i]
            is_last_chunk = i == new_chunk_count - 1

            try:
                if completed:
                    # Finalize with update_message_partial
                    # Only last chunk gets generating=False on completion
                    await self.channel.client.update_message_partial(
                        chunk_id,
                        user_id=message.user_id,
                        set={
                            "text": chunk_text,
                            "generating": False if is_last_chunk else False,
                            "chunk_group": message.id,
                            "chunk_index": i,
                            "total_chunks": new_chunk_count,
                        },
                    )
                    logger.debug(
                        f"Finalized chunk {i + 1}/{new_chunk_count} (ID: {chunk_id})"
                    )
                else:
                    # Update with ephemeral (still generating)
                    # Only last chunk gets generating=True when streaming
                    await self.channel.client.ephemeral_message_update(
                        chunk_id,
                        user_id=message.user_id,
                        set={
                            "text": chunk_text,
                            "generating": True if is_last_chunk else False,
                            "chunk_group": message.id,
                            "chunk_index": i,
                            "total_chunks": new_chunk_count,
                        },
                    )
                    logger.debug(
                        f"Updated chunk {i + 1}/{new_chunk_count} (ID: {chunk_id})"
                    )
            except Exception as e:
                logger.error(f"Failed to update chunk {i} (ID: {chunk_id}): {e}")
                # Don't re-raise - message is in memory

        # Create new chunks if content grew
        if new_chunk_count > old_chunk_count:
            for i in range(old_chunk_count, new_chunk_count):
                chunk_text = new_chunks[i]
                is_last_chunk = i == new_chunk_count - 1

                request = MessageRequest(
                    text=chunk_text,
                    user_id=message.user_id,
                    custom={
                        "generating": not completed if is_last_chunk else False,
                        "chunk_group": message.id,
                        "chunk_index": i,
                        "total_chunks": new_chunk_count,
                    },
                )

                try:
                    response = await self.channel.send_message(request)
                    chunk_id = response.data.message.id
                    state.backend_message_ids.append(chunk_id)
                    logger.debug(
                        f"Created new chunk {i + 1}/{new_chunk_count} (ID: {chunk_id})"
                    )
                except Exception as e:
                    logger.error(f"Failed to create new chunk {i}: {e}")
                    raise

        # Delete extra chunks if content shrank
        elif new_chunk_count < old_chunk_count:
            for i in range(new_chunk_count, old_chunk_count):
                chunk_id = state.backend_message_ids[i]
                try:
                    # Use client.delete_message, not channel.delete_message
                    await self.channel.client.delete_message(chunk_id, hard=True)
                    logger.debug(f"Deleted chunk {i + 1} (ID: {chunk_id})")
                except Exception as e:
                    logger.warning(f"Failed to delete chunk {i} (ID: {chunk_id}): {e}")

            # Remove from tracking
            state.backend_message_ids = state.backend_message_ids[:new_chunk_count]

    def _smart_chunk(self, text: str, max_size: int) -> List[str]:
        """Chunk text intelligently, respecting markdown structures.

        Best-effort approach to avoid breaking:
        - Code blocks (```)
        - Lists
        - Paragraphs

        Args:
            text: Text to chunk
            max_size: Maximum characters per chunk

        Returns:
            List of text chunks
        """
        if len(text) <= max_size:
            return [text]

        # Special case: text with no newlines (force split at max_size)
        if "\n" not in text:
            return self._force_split(text, max_size)

        chunks = []
        current_chunk = ""

        # Track markdown state
        in_code_block = False
        code_block_buffer = ""

        lines = text.split("\n")

        for line in lines:
            # Detect code block boundaries
            if line.strip().startswith("```"):
                if not in_code_block:
                    # Starting code block
                    in_code_block = True
                    code_block_buffer = line + "\n"
                else:
                    # Ending code block
                    in_code_block = False
                    code_block_buffer += line + "\n"

                    # Try to fit code block in current chunk
                    if len(current_chunk) + len(code_block_buffer) <= max_size:
                        current_chunk += code_block_buffer
                    elif len(code_block_buffer) <= max_size:
                        # Code block fits alone, start new chunk
                        if current_chunk:
                            chunks.append(current_chunk.rstrip())
                        current_chunk = code_block_buffer
                    else:
                        # Code block too large, must split it
                        if current_chunk:
                            chunks.append(current_chunk.rstrip())
                            current_chunk = ""

                        # Split code block at newlines
                        cb_chunks = self._split_large_block(code_block_buffer, max_size)
                        chunks.extend(cb_chunks[:-1])
                        current_chunk = cb_chunks[-1] if cb_chunks else ""

                    code_block_buffer = ""
                continue

            # Inside code block - accumulate
            if in_code_block:
                code_block_buffer += line + "\n"
                continue

            # Regular line - check if we should chunk
            line_with_newline = line + "\n"

            # Check if adding this line would exceed limit
            if len(current_chunk) + len(line_with_newline) > max_size:
                # Try to break at paragraph (double newline) or list boundary
                if current_chunk:
                    chunks.append(current_chunk.rstrip())
                    current_chunk = line_with_newline
            else:
                current_chunk += line_with_newline

        # Handle any remaining buffered content
        if in_code_block and code_block_buffer:
            # Unclosed code block - include as-is
            if len(current_chunk) + len(code_block_buffer) <= max_size:
                current_chunk += code_block_buffer
            else:
                if current_chunk:
                    chunks.append(current_chunk.rstrip())
                current_chunk = code_block_buffer

        if current_chunk:
            chunks.append(current_chunk.rstrip())

        return chunks if chunks else [text]  # Fallback to original text

    def _split_large_block(self, block: str, max_size: int) -> List[str]:
        """Split a large block (e.g., huge code block) at newline boundaries.

        Args:
            block: Large text block to split
            max_size: Maximum size per chunk

        Returns:
            List of chunks
        """
        if len(block) <= max_size:
            return [block]

        chunks = []
        lines = block.split("\n")
        current = ""

        for line in lines:
            if len(current) + len(line) + 1 > max_size:
                if current:
                    chunks.append(current)
                # If single line exceeds max_size, include it anyway
                current = line + "\n"
            else:
                current += line + "\n"

        if current:
            chunks.append(current)

        return chunks

    def _force_split(self, text: str, max_size: int) -> List[str]:
        """Force split text at max_size boundaries (no newlines available).

        Used for text without any newlines (e.g., long URLs, continuous text).

        Args:
            text: Text to split
            max_size: Maximum characters per chunk

        Returns:
            List of chunks
        """
        if len(text) <= max_size:
            return [text]

        chunks = []
        for i in range(0, len(text), max_size):
            chunks.append(text[i : i + max_size])

        return chunks
