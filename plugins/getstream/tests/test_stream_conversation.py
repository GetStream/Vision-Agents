import logging
from random import shuffle

import pytest
import uuid
import asyncio
from unittest.mock import Mock, AsyncMock
from dotenv import load_dotenv

from getstream.models import MessageRequest, ChannelInput, MessagePaginationParams
from getstream import AsyncStream

from vision_agents.core.agents.conversation import Message
from vision_agents.plugins.getstream.stream_conversation import StreamConversation

logger = logging.getLogger(__name__)

load_dotenv()


class TestStreamConversation:
    """Test suite for StreamConversation with unified API."""
    
    @pytest.fixture
    def mock_channel(self):
        """Create a mock Channel."""
        channel = Mock()
        channel.channel_type = "messaging"
        channel.channel_id = "test-channel-123"
        
        # Mock the client
        channel.client = Mock()
        
        # Create async mocks for client methods
        channel.client.update_message_partial = AsyncMock(return_value=Mock())
        channel.client.ephemeral_message_update = AsyncMock(return_value=Mock())
        
        # Mock send_message response
        mock_response = Mock()
        mock_response.data.message.id = "stream-message-123"
        mock_response.data.message.type = "regular"
        
        # Create async mock for send_message
        channel.send_message = AsyncMock(return_value=mock_response)
        
        return channel
    
    @pytest.fixture
    def stream_conversation(self, mock_channel):
        """Create a StreamConversation instance with mocked dependencies."""
        instructions = "You are a helpful assistant."
        messages = []
        conversation = StreamConversation(
            instructions=instructions,
            messages=messages,
            channel=mock_channel
        )
        return conversation
    
    @pytest.mark.asyncio
    async def test_send_message_simple(self, stream_conversation, mock_channel):
        """Test send_message convenience method."""
        message = await stream_conversation.send_message(
            role="user",
            user_id="user123",
            content="Hello",
        )
        
        # Verify message was added
        assert len(stream_conversation.messages) == 1
        assert stream_conversation.messages[0].content == "Hello"
        assert stream_conversation.messages[0].role == "user"
        assert stream_conversation.messages[0].user_id == "user123"
        
        # Verify Stream API was called
        mock_channel.send_message.assert_called_once()
        call_args = mock_channel.send_message.call_args
        request = call_args[0][0]
        assert isinstance(request, MessageRequest)
        assert request.text == "Hello"
        assert request.user_id == "user123"
        assert request.custom.get("generating") == False  # completed=True by default
    
    @pytest.mark.asyncio
    async def test_upsert_simple_message(self, stream_conversation, mock_channel):
        """Test adding a simple non-streaming message with upsert."""
        message = await stream_conversation.upsert_message(
            role="user",
            user_id="user123",
            content="Hello",
            completed=True,
        )
        
        # Verify message was added
        assert len(stream_conversation.messages) == 1
        assert stream_conversation.messages[0].content == "Hello"
        assert stream_conversation.messages[0].role == "user"
        assert stream_conversation.messages[0].user_id == "user123"
        
        # Verify Stream API was called
        mock_channel.send_message.assert_called_once()
        call_args = mock_channel.send_message.call_args
        request = call_args[0][0]
        assert isinstance(request, MessageRequest)
        assert request.text == "Hello"
        assert request.user_id == "user123"
    
    @pytest.mark.asyncio
    async def test_upsert_streaming_deltas(self, stream_conversation, mock_channel):
        """Test streaming message with deltas."""
        msg_id = str(uuid.uuid4())
        
        # Delta 1
        await stream_conversation.upsert_message(
            role="assistant",
            user_id="agent",
            content="Hello",
            message_id=msg_id,
            content_index=0,
            completed=False,
        )
        
        assert len(stream_conversation.messages) == 1
        assert stream_conversation.messages[0].content == "Hello"
        mock_channel.send_message.assert_called_once()
        
        # Delta 2
        mock_channel.send_message.reset_mock()
        await stream_conversation.upsert_message(
            role="assistant",
            user_id="agent",
            content=" world",
            message_id=msg_id,
            content_index=1,
            completed=False,
        )
        
        assert len(stream_conversation.messages) == 1
        assert stream_conversation.messages[0].content == "Hello world"
        # Should call ephemeral update, not send_message
        mock_channel.send_message.assert_not_called()
        mock_channel.client.ephemeral_message_update.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_upsert_streaming_completion(self, stream_conversation, mock_channel):
        """Test streaming message followed by completion."""
        msg_id = str(uuid.uuid4())
        
        # Streaming deltas
        await stream_conversation.upsert_message(
            role="assistant",
            user_id="agent",
            content="Hello",
            message_id=msg_id,
            content_index=0,
            completed=False,
        )
        
        await stream_conversation.upsert_message(
            role="assistant",
            user_id="agent",
            content=" world",
            message_id=msg_id,
            content_index=1,
            completed=False,
        )
        
        # Completion - replace with final text
        await stream_conversation.upsert_message(
            role="assistant",
            user_id="agent",
            content="Hello world!",
            message_id=msg_id,
            completed=True,
            replace=True,
        )
        
        # Should have only 1 message
        assert len(stream_conversation.messages) == 1
        assert stream_conversation.messages[0].content == "Hello world!"
        
        # Should call update_message_partial for completion
        mock_channel.client.update_message_partial.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_upsert_out_of_order_deltas(self, stream_conversation, mock_channel):
        """Test that out-of-order deltas are buffered correctly."""
        msg_id = str(uuid.uuid4())
        
        # Send deltas out of order
        await stream_conversation.upsert_message(
            role="assistant", user_id="agent",
            content=" world", message_id=msg_id, content_index=1, completed=False
        )
        
        # Message should exist but only have content when index 0 arrives
        assert len(stream_conversation.messages) == 1
        assert stream_conversation.messages[0].content == ""  # Waiting for index 0
        
        # Now send index 0
        await stream_conversation.upsert_message(
            role="assistant", user_id="agent",
            content="Hello", message_id=msg_id, content_index=0, completed=False
        )
        
        # Now it should have both
        assert stream_conversation.messages[0].content == "Hello world"
    
    @pytest.mark.asyncio
    async def test_upsert_replace_vs_append(self, stream_conversation, mock_channel):
        """Test replace vs append behavior."""
        msg_id = str(uuid.uuid4())
        
        # Create message
        await stream_conversation.upsert_message(
            role="assistant", user_id="agent",
            content="Hello", message_id=msg_id, completed=False
        )
        
        # Append
        await stream_conversation.upsert_message(
            role="assistant", user_id="agent",
            content=" world", message_id=msg_id, completed=False, replace=False
        )
        
        assert stream_conversation.messages[0].content == "Hello world"
        
        # Replace
        await stream_conversation.upsert_message(
            role="assistant", user_id="agent",
            content="Goodbye", message_id=msg_id, completed=True, replace=True
        )
        
        assert stream_conversation.messages[0].content == "Goodbye"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_streaming_deltas_then_completion_integration():
    """Integration test: streaming deltas followed by completion."""
    channel_id = f"test-channel-{uuid.uuid4()}"
    chat_client = AsyncStream().chat
    channel = chat_client.channel("messaging", channel_id)
    
    await channel.get_or_create(
        data=ChannelInput(created_by_id="test-user"),
    )
    
    conversation = StreamConversation(
        instructions="Test conversation",
        messages=[],
        channel=channel
    )
    
    msg_id = str(uuid.uuid4())
    
    # Send streaming deltas
    logger.info("Sending deltas...")
    await conversation.upsert_message(
        role="assistant", user_id="agent",
        content="Hello", message_id=msg_id, content_index=0, completed=False
    )
    
    await conversation.upsert_message(
        role="assistant", user_id="agent",
        content=" world", message_id=msg_id, content_index=1, completed=False
    )
    
    await conversation.upsert_message(
        role="assistant", user_id="agent",
        content="!", message_id=msg_id, content_index=2, completed=False
    )
    
    # Complete the message
    logger.info("Completing message...")
    await conversation.upsert_message(
        role="assistant", user_id="agent",
        content="Hello world!", message_id=msg_id, completed=True, replace=True
    )
    
    # Verify only 1 message in memory
    assert len(conversation.messages) == 1
    assert conversation.messages[0].content == "Hello world!"
    
    # Verify only 1 message in Stream
    response = await channel.get_or_create(state=True, messages=MessagePaginationParams(limit=10))
    assert len(response.data.messages) == 1
    assert response.data.messages[0].text == "Hello world!"
    
    logger.info("✅ Test passed: Only 1 message created")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_completion_before_deltas_integration():
    """Integration test: completion arrives before deltas (race condition)."""
    channel_id = f"test-channel-{uuid.uuid4()}"
    chat_client = AsyncStream().chat
    channel = chat_client.channel("messaging", channel_id)
    
    await channel.get_or_create(
        data=ChannelInput(created_by_id="test-user"),
    )
    
    conversation = StreamConversation(
        instructions="Test conversation",
        messages=[],
        channel=channel
    )
    
    msg_id = str(uuid.uuid4())
    full_text = "Hello world!"
    
    # Completion arrives first
    logger.info("Completion arrives first...")
    await conversation.upsert_message(
        role="assistant", user_id="agent",
        content=full_text, message_id=msg_id, completed=True, replace=True
    )
    
    assert len(conversation.messages) == 1
    assert conversation.messages[0].content == full_text
    
    # Deltas arrive late (should be no-op since message is completed)
    logger.info("Late deltas arrive...")
    await conversation.upsert_message(
        role="assistant", user_id="agent",
        content="Hello", message_id=msg_id, content_index=0, completed=False
    )
    
    # Should still be only 1 message with full text (deltas ignored after completion)
    assert len(conversation.messages) == 1
    assert conversation.messages[0].content == full_text
    
    # Verify only 1 message in Stream
    response = await channel.get_or_create(state=True, messages=MessagePaginationParams(limit=10))
    assert len(response.data.messages) == 1
    assert response.data.messages[0].text == full_text
    
    logger.info("✅ Test passed: Only 1 message, late deltas ignored")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_out_of_order_fragments_integration():
    """Integration test: out-of-order delta fragments."""
    channel_id = f"test-channel-{uuid.uuid4()}"
    chat_client = AsyncStream().chat
    channel = chat_client.channel("messaging", channel_id)
    
    await channel.get_or_create(
        data=ChannelInput(created_by_id="test-user"),
    )
    
    conversation = StreamConversation(
        instructions="Test conversation",
        messages=[],
        channel=channel
    )
    
    msg_id = str(uuid.uuid4())
    
    # Send fragments out of order
    chunks = [
        (0, "once"),
        (1, " upon"),
        (2, " a"),
        (3, " time"),
        (4, " in"),
        (5, " a"),
        (6, " galaxy"),
        (7, " far"),
        (8, " far"),
        (9, " away"),
    ]
    
    shuffle(chunks)
    
    for idx, txt in chunks:
        await conversation.upsert_message(
            role="assistant", user_id="agent",
            content=txt, message_id=msg_id, content_index=idx, completed=False
        )
    
    # Complete the message
    await conversation.upsert_message(
        role="assistant", user_id="agent",
        content="once upon a time in a galaxy far far away",
        message_id=msg_id, completed=True, replace=True
    )
    
    # Verify only 1 message
    assert len(conversation.messages) == 1
    
    # Verify in Stream
    response = await channel.get_or_create(state=True, messages=MessagePaginationParams(limit=10))
    assert len(response.data.messages) == 1
    assert response.data.messages[0].text == "once upon a time in a galaxy far far away"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_race_condition_delta_and_completion_concurrent():
    """Test the exact race condition from the bug report.
    
    Simulates delta and completion events arriving at nearly the same time
    (both checking state before either completes).
    """
    channel_id = f"test-channel-{uuid.uuid4()}"
    chat_client = AsyncStream().chat
    channel = chat_client.channel("messaging", channel_id)
    
    await channel.get_or_create(
        data=ChannelInput(created_by_id="test-user"),
    )
    
    conversation = StreamConversation(
        instructions="Test conversation",
        messages=[],
        channel=channel
    )
    
    msg_id = str(uuid.uuid4())
    
    # Send delta and completion concurrently (race condition)
    logger.info("Sending delta and completion concurrently...")
    await asyncio.gather(
        # Delta arrives
        conversation.upsert_message(
            role="assistant", user_id="agent",
            content="The", message_id=msg_id, content_index=0, completed=False
        ),
        # Completion arrives at same time
        conversation.upsert_message(
            role="assistant", user_id="agent",
            content="The old lighthouse keeper...",
            message_id=msg_id, completed=True, replace=True
        ),
    )
    
    # Should have only 1 message (not 2!)
    assert len(conversation.messages) == 1
    assert conversation.messages[0].content == "The old lighthouse keeper..."
    
    # Verify only 1 message in Stream
    response = await channel.get_or_create(state=True, messages=MessagePaginationParams(limit=10))
    assert len(response.data.messages) == 1, f"Expected 1 message, got {len(response.data.messages)}"
    assert response.data.messages[0].text == "The old lighthouse keeper..."
    
    logger.info("✅ Race condition test passed: Only 1 message created")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_messages():
    """Test multiple concurrent streaming messages."""
    channel_id = f"test-channel-{uuid.uuid4()}"
    chat_client = AsyncStream().chat
    channel = chat_client.channel("messaging", channel_id)
    
    await channel.get_or_create(
        data=ChannelInput(created_by_id="test-user"),
    )
    
    conversation = StreamConversation(
        instructions="Test conversation",
        messages=[],
        channel=channel
    )
    
    msg_id_1 = str(uuid.uuid4())
    msg_id_2 = str(uuid.uuid4())
    
    # Stream two messages concurrently
    await asyncio.gather(
        conversation.upsert_message(
            role="user", user_id="user1",
            content="Question 1", message_id=msg_id_1, completed=True
        ),
        conversation.upsert_message(
            role="assistant", user_id="agent",
            content="Answer 1", message_id=msg_id_2, completed=True
        ),
    )
    
    # Should have 2 messages
    assert len(conversation.messages) == 2
    
    # Verify in Stream
    response = await channel.get_or_create(state=True, messages=MessagePaginationParams(limit=10))
    assert len(response.data.messages) == 2
