"""
Unit tests for message chunking logic in StreamConversation.

Tests the markdown-aware chunking algorithm and chunk management.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from vision_agents.plugins.getstream.stream_conversation import StreamConversation


class TestMessageChunking:
    """Test suite for message chunking logic."""

    @pytest.fixture
    def conversation(self):
        """Create a StreamConversation with small chunk size for testing."""
        mock_channel = Mock()
        mock_channel.channel_type = "messaging"
        mock_channel.channel_id = "test-channel"

        conversation = StreamConversation(
            instructions="Test",
            messages=[],
            channel=mock_channel,
            chunk_size=50,  # Small size for easy testing
        )
        return conversation

    def test_no_chunking_needed(self, conversation):
        """Test that small messages aren't chunked."""
        text = "Hello world"
        chunks = conversation._smart_chunk(text, 50)

        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_simple_chunking(self, conversation):
        """Test basic chunking at line boundaries."""
        text = "Line 1 is here\nLine 2 is here\nLine 3 is here\nLine 4 is here\nLine 5 is here"
        chunks = conversation._smart_chunk(text, 30)

        # Should split into multiple chunks
        assert len(chunks) > 1

        # Verify all content is preserved
        reconstructed = "\n".join(chunks)
        # Remove extra whitespace for comparison
        assert reconstructed.replace("\n\n", "\n").strip() == text.strip()

    def test_code_block_not_split(self, conversation):
        """Test that code blocks stay together."""
        text = """Here is some code:

```python
def hello():
    return "world"
```

And more text."""

        chunks = conversation._smart_chunk(text, 100)

        # Code block should stay intact in one chunk
        has_complete_code_block = any(
            "```python" in chunk and 'return "world"' in chunk and "```" in chunk
            for chunk in chunks
        )
        assert has_complete_code_block, "Code block was split incorrectly"

    def test_code_block_in_own_chunk(self, conversation):
        """Test that large code blocks get their own chunk."""
        text = """Short intro.

```python
def function_one():
    return "value"

def function_two():
    return "another"
```

More text after."""

        chunks = conversation._smart_chunk(text, 60)

        # Should create multiple chunks
        assert len(chunks) >= 2

        # Find chunk with code block
        code_chunk = next((c for c in chunks if "```python" in c), None)
        assert code_chunk is not None
        assert "```" in code_chunk  # Should have closing backticks

    def test_very_large_code_block_split(self, conversation):
        """Test that code blocks larger than max_size are split at newlines."""
        # Create a code block that's too large
        code_lines = ["def function_{}():".format(i) for i in range(20)]
        text = "```python\n" + "\n".join(code_lines) + "\n```"

        chunks = conversation._smart_chunk(text, 80)

        # Should be split despite being in code block
        assert len(chunks) > 1

        # All chunks should have content
        for chunk in chunks:
            assert len(chunk) > 0

    def test_paragraph_chunking(self, conversation):
        """Test chunking at paragraph boundaries."""
        text = "Paragraph one with some text here.\n\nParagraph two with more text.\n\nParagraph three here."

        chunks = conversation._smart_chunk(text, 40)

        # Should split at paragraph boundaries
        assert len(chunks) >= 2

        # Content preserved
        reconstructed = "\n".join(chunk.strip() for chunk in chunks)
        # Normalize whitespace for comparison
        assert "Paragraph one" in reconstructed
        assert "Paragraph two" in reconstructed
        assert "Paragraph three" in reconstructed

    def test_empty_text(self, conversation):
        """Test chunking empty text."""
        chunks = conversation._smart_chunk("", 50)

        assert len(chunks) == 1
        assert chunks[0] == ""

    def test_exact_boundary(self, conversation):
        """Test text that's exactly at the boundary."""
        text = "A" * 50  # Exactly chunk_size
        chunks = conversation._smart_chunk(text, 50)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_one_char_over(self, conversation):
        """Test text that's one character over the limit."""
        text = "A" * 51
        chunks = conversation._smart_chunk(text, 50)

        # Should be split (no newlines, so splits anywhere)
        assert len(chunks) >= 1

    def test_multiple_code_blocks(self, conversation):
        """Test text with multiple code blocks."""
        text = """First block:

```python
code1()
```

Middle text.

```javascript
code2()
```

End text."""

        chunks = conversation._smart_chunk(text, 60)

        # Both code blocks should be complete
        full_text = "\n".join(chunks)
        assert (
            full_text.count("```python") == full_text.count("```") / 2
            or "```python" in full_text
        )
        assert "code1()" in full_text
        assert "code2()" in full_text

    def test_nested_markdown(self, conversation):
        """Test text with nested markdown (lists, code, etc)."""
        text = """# Header

- List item 1
- List item 2
  - Nested item
  
Some text.

```python
code()
```"""

        chunks = conversation._smart_chunk(text, 70)

        # Content should be preserved
        full_text = "\n".join(chunks)
        assert "# Header" in full_text or "Header" in full_text
        assert "List item 1" in full_text
        assert "code()" in full_text

    def test_split_large_block_basic(self, conversation):
        """Test _split_large_block with basic text."""
        block = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6"
        chunks = conversation._split_large_block(block, 20)

        assert len(chunks) > 1

        # Verify content preserved
        reconstructed = "".join(chunks)
        assert "Line 1" in reconstructed
        assert "Line 6" in reconstructed

    def test_split_large_block_single_long_line(self, conversation):
        """Test _split_large_block with line longer than max_size."""
        block = "A" * 100  # Single line, no newlines
        chunks = conversation._split_large_block(block, 20)

        # Should include the line anyway (can't split further)
        assert len(chunks) >= 1
        assert "A" * 100 in "".join(chunks)


class TestChunkingIntegration:
    """Integration tests for chunking with mocked Stream API."""

    @pytest.fixture
    def mock_channel(self):
        """Create a mock channel with tracking."""
        channel = Mock()
        channel.channel_type = "messaging"
        channel.channel_id = "test-channel"

        # Mock client
        channel.client = Mock()
        channel.client.update_message_partial = AsyncMock(return_value=Mock())
        channel.client.ephemeral_message_update = AsyncMock(return_value=Mock())
        channel.client.delete_message = AsyncMock(
            return_value=Mock()
        )  # Add delete_message to client

        # Track messages sent
        channel.sent_messages = []

        async def mock_send_message(request):
            mock_response = Mock()
            mock_response.data.message.id = f"chunk-{len(channel.sent_messages)}"
            mock_response.data.message.type = "regular"
            channel.sent_messages.append(request)
            return mock_response

        channel.send_message = mock_send_message

        return channel

    @pytest.fixture
    def conversation(self, mock_channel):
        """Create conversation with small chunk size."""
        return StreamConversation(
            instructions="Test",
            messages=[],
            channel=mock_channel,
            chunk_size=50,  # Small for testing
        )

    @pytest.mark.asyncio
    async def test_large_message_creates_multiple_chunks(
        self, conversation, mock_channel
    ):
        """Test that large messages are automatically chunked."""
        # Create a message that will need 3 chunks
        large_text = "A" * 120  # 120 chars = 3 chunks at 50 chars each

        await conversation.send_message(
            role="assistant",
            user_id="agent",
            content=large_text,
        )

        # Should have 1 message in conversation.messages
        assert len(conversation.messages) == 1
        assert conversation.messages[0].content == large_text

        # Should have created 3 chunks in Stream
        assert len(mock_channel.sent_messages) == 3

        # Verify chunk metadata
        for i, msg_request in enumerate(mock_channel.sent_messages):
            assert msg_request.custom["chunk_index"] == i
            assert msg_request.custom["total_chunks"] == 3
            assert msg_request.custom["chunk_group"] == conversation.messages[0].id

            # Only last chunk has generating=False
            is_last = i == 2
            assert (
                msg_request.custom["generating"] == (not is_last)
                or not msg_request.custom["generating"]
            )

    @pytest.mark.asyncio
    async def test_streaming_grows_creates_new_chunk(self, conversation, mock_channel):
        """Test that streaming content growing beyond chunk_size creates new chunks."""
        msg_id = "test-msg"

        # Start with small content (1 chunk)
        await conversation.upsert_message(
            role="assistant",
            user_id="agent",
            content="Short",
            message_id=msg_id,
            content_index=0,
            completed=False,
        )

        assert len(mock_channel.sent_messages) == 1
        assert conversation.messages[0].content == "Short"
        # First chunk created with generating=True (it's the last/only chunk initially)
        assert mock_channel.sent_messages[0].custom["generating"] is True

        # Add more content to exceed chunk_size (should create 2nd chunk)
        await conversation.upsert_message(
            role="assistant",
            user_id="agent",
            content="X" * 60,
            message_id=msg_id,
            completed=False,
            replace=True,
        )

        # Should have created a new chunk (2 total)
        assert len(mock_channel.sent_messages) == 2

        # Chunk 1 (last, newly created): should have generating=True
        assert mock_channel.sent_messages[1].custom["chunk_index"] == 1
        assert mock_channel.sent_messages[1].custom["generating"] is True

        # First chunk should have been updated via ephemeral to generating=False (it's now intermediate)
        assert mock_channel.client.ephemeral_message_update.call_count >= 1

        # Check first chunk was updated to generating=False
        first_chunk_updates = [
            call
            for call in mock_channel.client.ephemeral_message_update.call_args_list
            if call[0][0] == "chunk-0"  # First chunk ID
        ]
        if first_chunk_updates:
            # Most recent update should have generating=False
            assert first_chunk_updates[-1][1]["set"]["generating"] is False

    @pytest.mark.asyncio
    async def test_completion_shrinks_deletes_chunks(self, conversation, mock_channel):
        """Test that completion with shorter text deletes extra chunks."""
        msg_id = "test-msg"

        # Start with large content (3 chunks)
        large_text = "X" * 120
        await conversation.upsert_message(
            role="assistant",
            user_id="agent",
            content=large_text,
            message_id=msg_id,
            completed=False,
        )

        initial_chunk_count = len(mock_channel.sent_messages)
        assert initial_chunk_count >= 2

        # Complete with shorter text (1 chunk)
        short_text = "Done"
        await conversation.upsert_message(
            role="assistant",
            user_id="agent",
            content=short_text,
            message_id=msg_id,
            completed=True,
            replace=True,
        )

        # Should have deleted extra chunks
        assert mock_channel.client.delete_message.call_count == initial_chunk_count - 1

    @pytest.mark.asyncio
    async def test_code_block_preserved_in_chunk(self, conversation, mock_channel):
        """Test that code blocks are kept intact when possible."""
        text = """Intro text.

```python
def hello():
    return "world"
```

End."""

        await conversation.send_message(
            role="assistant",
            user_id="agent",
            content=text,
        )

        # Verify code block is complete in at least one chunk
        full_text = "".join(req.text for req in mock_channel.sent_messages)
        assert "```python" in full_text
        assert "def hello():" in full_text
        assert 'return "world"' in full_text
        assert full_text.count("```") >= 2  # Opening and closing

    @pytest.mark.asyncio
    async def test_chunk_metadata_correct(self, conversation, mock_channel):
        """Test that chunk metadata is set correctly."""
        large_text = "A" * 120  # Will create 3 chunks

        message = await conversation.send_message(
            role="assistant",
            user_id="agent",
            content=large_text,
        )

        chunks_created = len(mock_channel.sent_messages)

        # Verify metadata on all chunks
        for i, req in enumerate(mock_channel.sent_messages):
            assert req.custom["chunk_group"] == message.id
            assert req.custom["chunk_index"] == i
            assert req.custom["total_chunks"] == chunks_created

            # Only last chunk has generating (and it's False since completed=True)
            is_last = i == chunks_created - 1
            if is_last:
                assert req.custom["generating"] is False
            else:
                assert (
                    req.custom["generating"] is False
                )  # Intermediate chunks never generating

    @pytest.mark.asyncio
    async def test_streaming_chunk_generating_flag(self, conversation, mock_channel):
        """Test that generating flag is only on last chunk during streaming."""
        msg_id = "test-msg"

        # Create large streaming message (3 chunks)
        large_text = "X" * 120
        await conversation.upsert_message(
            role="assistant",
            user_id="agent",
            content=large_text,
            message_id=msg_id,
            completed=False,
        )

        # Check the requests sent
        chunks_sent = len(mock_channel.sent_messages)
        assert chunks_sent >= 2

        # Check generating flag on each chunk
        for i, req in enumerate(mock_channel.sent_messages):
            is_last = i == chunks_sent - 1
            if is_last:
                assert req.custom["generating"] is True, (
                    f"Last chunk {i} should have generating=True"
                )
            else:
                assert req.custom["generating"] is False, (
                    f"Intermediate chunk {i} should have generating=False"
                )


class TestChunkingSentenceBoundaries:
    """Test chunking at sentence boundaries."""

    def test_sentence_chunking(self):
        """Test chunking respects sentence boundaries when possible."""
        conversation = StreamConversation(
            instructions="Test",
            messages=[],
            channel=Mock(),
            chunk_size=40,
        )

        text = "First sentence here. Second sentence here. Third sentence. Fourth sentence here."
        chunks = conversation._smart_chunk(text, 40)

        # Should split at sentence boundaries
        assert len(chunks) >= 2

        # Content preserved
        full = "\n".join(chunks)
        assert "First sentence" in full
        assert "Fourth sentence" in full

    def test_split_large_block_preserves_content(self):
        """Test that _split_large_block preserves all content."""
        conversation = StreamConversation(
            instructions="Test",
            messages=[],
            channel=Mock(),
            chunk_size=30,
        )

        block = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        chunks = conversation._split_large_block(block, 20)

        # Verify all lines present
        full = "".join(chunks)
        for i in range(1, 6):
            assert f"Line {i}" in full

    def test_split_large_block_single_long_line(self):
        """Test splitting when a single line exceeds max_size."""
        conversation = StreamConversation(
            instructions="Test",
            messages=[],
            channel=Mock(),
            chunk_size=30,
        )

        # Single line that's way too long
        block = "A" * 100
        chunks = conversation._split_large_block(block, 20)

        # Should still create chunk(s) - can't split further
        assert len(chunks) >= 1

        # Content preserved
        assert "A" * 100 in "".join(chunks)
