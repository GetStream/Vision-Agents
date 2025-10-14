import logging
from random import shuffle

import pytest
import uuid
import asyncio
from unittest.mock import Mock, AsyncMock
from dotenv import load_dotenv

from getstream.models import MessageRequest, ChannelInput, MessagePaginationParams
from getstream import AsyncStream

from vision_agents.core.agents.conversation import (
    Message,
)
from vision_agents.plugins.getstream.stream_conversation import StreamConversation

logger = logging.getLogger(__name__)

load_dotenv()

class TestStreamConversation:
    """Test suite for StreamConversation class."""
    
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
        
        # Create async mock for send_message
        channel.send_message = AsyncMock(return_value=mock_response)
        
        return channel
    
    @pytest.fixture
    def stream_conversation(self, mock_channel):
        """Create a StreamConversation instance with mocked dependencies."""
        instructions = "You are a helpful assistant."
        messages = [
            Message(
                original=None,
                content="Hello",
                role="user",
                user_id="user1",
            )
        ]
        # Set IDs for messages
        for i, msg in enumerate(messages):
            msg.id = f"msg-{i}"
            
        conversation = StreamConversation(
            instructions=instructions,
            messages=messages,
            channel=mock_channel
        )
        
        # Pre-populate some stream IDs for testing
        conversation.internal_ids_to_stream_ids = {
            "msg-0": "stream-msg-0"
        }
        
        return conversation
    
    @pytest.mark.asyncio
    async def test_initialization(self, stream_conversation, mock_channel):
        """Test StreamConversation initialization."""
        assert stream_conversation.channel == mock_channel
        assert isinstance(stream_conversation.internal_ids_to_stream_ids, dict)
        assert len(stream_conversation.messages) == 1
    
    @pytest.mark.asyncio
    async def test_add_message(self, stream_conversation, mock_channel):
        """Test adding a message to the stream with default completed=True."""
        new_message = Message(
            original=None,
            content="Test message",
            role="user",
            user_id="user123"
        )
        new_message.id = "new-msg-id"
        
        await stream_conversation.add_message(new_message)
        
        # Verify message was added locally immediately
        assert len(stream_conversation.messages) == 2
        assert stream_conversation.messages[-1] == new_message
        
        # Verify Stream API was called
        mock_channel.send_message.assert_called_once()
        call_args = mock_channel.send_message.call_args

        request = call_args[0][0]
        assert isinstance(request, MessageRequest)
        assert request.text == "Test message"
        assert request.user_id == "user123"
        
        # Verify ID mapping was stored
        assert "new-msg-id" in stream_conversation.internal_ids_to_stream_ids
        assert stream_conversation.internal_ids_to_stream_ids["new-msg-id"] == "stream-message-123"
    
    @pytest.mark.asyncio
    async def test_add_message_with_completed_false(self, stream_conversation, mock_channel):
        """Test adding a message with completed=False (still generating)."""
        # Reset mocks
        mock_channel.send_message.reset_mock()
        mock_channel.client.ephemeral_message_update.reset_mock()
        mock_channel.client.update_message_partial.reset_mock()
        
        new_message = Message(
            original=None,
            content="Generating message",
            role="assistant",
            user_id="assistant"
        )
        new_message.id = "gen-msg-id"
        
        await stream_conversation.add_message(new_message, completed=False)
        
        # Verify message was added locally
        assert len(stream_conversation.messages) == 2
        assert stream_conversation.messages[-1] == new_message
        
        # Verify Stream API was called
        mock_channel.send_message.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_message_existing(self, stream_conversation, mock_channel):
        """Test updating an existing message by appending content."""
        # Update existing message by appending (replace_content=False, completed=False)
        await stream_conversation.update_message(
            message_id="msg-0",
            input_text=" additional text",
            user_id="user1",
            replace_content=False,
            completed=False
        )
        
        # Verify message content was appended immediately
        assert stream_conversation.messages[0].content == "Hello additional text"
        
        # Verify Stream API was called with ephemeral_message_update (not completed)
        mock_channel.client.ephemeral_message_update.assert_called_once()
        call_args = mock_channel.client.ephemeral_message_update.call_args
        assert call_args[0][0] == "stream-msg-0"  # stream message ID
        assert call_args[1]["user_id"] == "user1"
        assert call_args[1]["set"]["text"] == "Hello additional text"
        assert call_args[1]["set"]["generating"] is True  # not completed = still generating
    
    @pytest.mark.asyncio
    async def test_update_message_replace(self, stream_conversation, mock_channel):
        """Test replacing message content (replace_content=True)."""
        # Mock update_message_partial for completed messages
        mock_channel.client.update_message_partial = AsyncMock(return_value=Mock())
        
        await stream_conversation.update_message(
            message_id="msg-0",
            input_text="Replaced content",
            user_id="user1",
            replace_content=True,
            completed=True
        )
        
        # Verify message content was replaced
        assert stream_conversation.messages[0].content == "Replaced content"
        
        # Verify Stream API was called with update_message_partial (completed)
        mock_channel.client.update_message_partial.assert_called_once()
        call_args = mock_channel.client.update_message_partial.call_args
        assert call_args[0][0] == "stream-msg-0"  # stream message ID
        assert call_args[1]["user_id"] == "user1"
        assert call_args[1]["set"]["text"] == "Replaced content"
        assert call_args[1]["set"]["generating"] is False  # completed = not generating
    
    @pytest.mark.asyncio
    async def test_update_message_not_found(self, stream_conversation, mock_channel):
        """Test updating a non-existent message creates a new one."""
        # Reset the send_message mock for this test
        mock_channel.send_message.reset_mock()
        
        await stream_conversation.update_message(
            message_id="non-existent-id",
            input_text="New message content",
            user_id="user2",
            replace_content=True,
            completed=False
        )
        
        # Should have added a new message
        assert len(stream_conversation.messages) == 2
        
        # Verify the new message was created correctly
        new_msg = stream_conversation.messages[-1]
        assert new_msg.id == "non-existent-id"
        assert new_msg.content == "New message content"
        assert new_msg.user_id == "user2"
        
        # Verify send_message was called (not update)
        mock_channel.send_message.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_message_completed_vs_generating(self, stream_conversation, mock_channel):
        """Test that completed=True calls update_message_partial and completed=False calls ephemeral_message_update."""
        # Mock update_message_partial for completed messages
        mock_channel.client.update_message_partial = AsyncMock(return_value=Mock())
        
        # Test with completed=False (still generating)
        await stream_conversation.update_message(
            message_id="msg-0",
            input_text=" in progress",
            user_id="user1",
            replace_content=False,
            completed=False
        )
        
        # Should call ephemeral_message_update
        mock_channel.client.ephemeral_message_update.assert_called()
        mock_channel.client.update_message_partial.assert_not_called()
        
        # Reset mocks
        mock_channel.client.ephemeral_message_update.reset_mock()
        
        # Test with completed=True
        await stream_conversation.update_message(
            message_id="msg-0",
            input_text="Final content",
            user_id="user1",
            replace_content=True,
            completed=True
        )
        
        # Should call update_message_partial
        mock_channel.client.update_message_partial.assert_called_once()
        mock_channel.client.ephemeral_message_update.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_update_message_no_stream_id(self, stream_conversation, mock_channel):
        """Test updating a message without a stream ID mapping."""
        # Add a message without stream ID mapping
        new_msg = Message(
            original=None,
            content="Test",
            role="user",
            user_id="user3"
        )
        new_msg.id = "unmapped-msg"
        stream_conversation.messages.append(new_msg)
        
        # Try to update it by appending
        await stream_conversation.update_message(
            message_id="unmapped-msg",
            input_text=" updated",
            user_id="user3",
            replace_content=False,
            completed=False
        )
        
        # Message should still be updated locally (with space handling)
        assert stream_conversation.messages[-1].content == "Test updated"
        
        # Since there's no stream_id mapping, the API call should be skipped
        # This is the expected behavior - we don't sync messages without stream IDs
        mock_channel.client.ephemeral_message_update.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_streaming_message_handle(self, stream_conversation, mock_channel):
        """Test streaming message with handle API."""
        # Reset mocks
        mock_channel.send_message.reset_mock()
        mock_channel.client.ephemeral_message_update.reset_mock()
        mock_channel.client.update_message_partial.reset_mock()
        
        # Start a streaming message
        handle = await stream_conversation.start_streaming_message(role="assistant", initial_content="Processing")
        
        # Verify message was added and marked as generating
        assert len(stream_conversation.messages) == 2
        assert stream_conversation.messages[-1].content == "Processing"
        assert stream_conversation.messages[-1].role == "assistant"
        assert handle.user_id == "assistant"
        
        # Verify send_message was called
        mock_channel.send_message.assert_called_once()

        # Reset for next operations
        mock_channel.client.ephemeral_message_update.reset_mock()
        
        # Append to the message
        await stream_conversation.append_to_message(handle, "...")
        assert stream_conversation.messages[-1].content == "Processing..."
        
        mock_channel.client.ephemeral_message_update.assert_called_once()
        
        # Replace the message
        await stream_conversation.replace_message(handle, "Complete response")
        assert stream_conversation.messages[-1].content == "Complete response"
        
        assert mock_channel.client.ephemeral_message_update.call_count == 2
        
        # Complete the message
        mock_channel.client.update_message_partial.reset_mock()
        await stream_conversation.complete_message(handle)
        
        mock_channel.client.update_message_partial.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_multiple_streaming_handles(self, stream_conversation, mock_channel):
        """Test multiple concurrent streaming messages with Stream API."""
        # Reset mocks
        mock_channel.send_message.reset_mock()
        mock_channel.client.ephemeral_message_update.reset_mock()
        
        # Mock different message IDs for each send
        mock_response1 = Mock()
        mock_response1.data.message.id = "stream-msg-1"
        mock_response2 = Mock()
        mock_response2.data.message.id = "stream-msg-2"
        mock_channel.send_message.side_effect = [mock_response1, mock_response2]
        
        # Start two streaming messages with empty initial content
        handle1 = await stream_conversation.start_streaming_message(role="user", user_id="user123", initial_content="")
        handle2 = await stream_conversation.start_streaming_message(role="assistant", initial_content="")
        
        assert len(stream_conversation.messages) == 3  # 1 initial + 2 new
        
        # Update them independently
        await stream_conversation.append_to_message(handle1, "Hello?")
        await stream_conversation.append_to_message(handle2, "Hi there!")
        
        # Find messages by their handles to verify correct updates
        msg1 = next(msg for msg in stream_conversation.messages if msg.id == handle1.message_id)
        msg2 = next(msg for msg in stream_conversation.messages if msg.id == handle2.message_id)
        
        assert msg1.content == "Hello?"
        assert msg2.content == "Hi there!"
        
        # Verify ephemeral updates were called for both
        assert mock_channel.client.ephemeral_message_update.call_count >= 2
        
        # Complete both
        await stream_conversation.complete_message(handle1)
        await stream_conversation.complete_message(handle2)
        
        # Verify update_message_partial was called for both completions
        assert mock_channel.client.update_message_partial.call_count == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_large_message_handling_fails():
    """Test that large messages (>1000 chars) fail as expected.
    
    This test should FAIL because the feature is currently broken.
    We want this test to fail to track the issue.
    """

    # Create a test channel
    channel_id = f"test-channel-{uuid.uuid4()}"
    chat_client = AsyncStream().chat
    channel = chat_client.channel("messaging", channel_id)
    
    # Create the channel in Stream
    await channel.get_or_create(
        data=ChannelInput(created_by_id="test-user"),
    )
    
    # Create a conversation
    conversation = StreamConversation(
        instructions="Test conversation",
        messages=[],
        channel=channel
    )
    
    large_content = "A" * 30000  # 30000 characters, well over the 1000 limit
    
    # This should work (create the message)
    message = Message(
        original=None,
        content=large_content,
        role="assistant",
        user_id="test-agent"
    )
    message.id = str(uuid.uuid4())
    
    # This should fail because the message is too large for Stream's limits
    with pytest.raises(Exception):
        await conversation.add_message(message)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_content_index():
    channel_id = f"test-channel-{uuid.uuid4()}"
    chat_client = AsyncStream().chat
    channel = chat_client.channel("messaging", channel_id)

    # Create the channel in Stream
    await channel.get_or_create(
        data=ChannelInput(created_by_id="test-user"),
    )

    # Create a conversation
    conversation = StreamConversation(
        instructions="Test conversation",
        messages=[],
        channel=channel
    )

    coros = []
    item_id = str(uuid.uuid4())

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
        coros += [conversation.upsert_streaming_message_handler(
            message_id=item_id,
            user_id="agent",
            role="assistant",
            content=txt,
            content_index=idx,
        )]
        print(txt)
        await asyncio.sleep(0.01)

    # asyncio is weird
    handlers = await asyncio.gather(*coros)
    await handlers[0].finalize()
    response = await channel.get_or_create(state=True, messages=MessagePaginationParams(limit=10))
    assert len(response.data.messages) == 1
    assert response.data.messages[0].text == "once upon a time in a galaxy far far away"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_streaming_deltas_then_finalize_no_duplicate():
    """Test that simulates Agent's LLM flow: deltas arrive, then completed event.
    
    This test should ensure only 1 message is created (not 2).
    Simulates the exact flow from agents.py:
    1. _handle_output_text_delta calls upsert_streaming_message_handler for each delta
    2. on_llm_response_sync_conversation finds the handler and calls update_message
    """
    channel_id = f"test-channel-{uuid.uuid4()}"
    chat_client = AsyncStream().chat
    channel = chat_client.channel("messaging", channel_id)
    
    # Create the channel in Stream
    await channel.get_or_create(
        data=ChannelInput(created_by_id="test-user"),
    )
    
    # Create a conversation
    conversation = StreamConversation(
        instructions="Test conversation",
        messages=[],
        channel=channel
    )
    
    item_id = str(uuid.uuid4())
    
    # Simulate delta events arriving (like _handle_output_text_delta)
    logger.info("Simulating delta events...")
    logger.info(f"Initial message count: {len(conversation.messages)}")
    
    # Delta 1
    handler = await conversation.upsert_streaming_message_handler(
        message_id=item_id,
        user_id="agent",
        role="assistant",
        content="Hello",
        content_index=0,
    )
    logger.info(f"After delta 1, message count: {len(conversation.messages)}, messages: {[m.content for m in conversation.messages]}")
    
    # Delta 2
    await conversation.upsert_streaming_message_handler(
        message_id=item_id,
        user_id="agent",
        role="assistant",
        content=" world",
        content_index=1,
    )
    logger.info(f"After delta 2, message count: {len(conversation.messages)}, messages: {[m.content for m in conversation.messages]}")
    
    # Delta 3
    await conversation.upsert_streaming_message_handler(
        message_id=item_id,
        user_id="agent",
        role="assistant",
        content="!",
        content_index=2,
    )
    logger.info(f"After delta 3, message count: {len(conversation.messages)}, messages: {[m.content for m in conversation.messages]}")
    
    # Check that only 1 message exists in conversation.messages so far
    assert len(conversation.messages) == 1, f"Expected 1 message after deltas, got {len(conversation.messages)}: {[m.content for m in conversation.messages]}"
    assert conversation.messages[0].content == "Hello world!", f"Expected 'Hello world!', got '{conversation.messages[0].content}'"
    
    # Simulate completed event (like on_llm_response_sync_conversation)
    logger.info("Simulating completed event...")
    existing_handler = await conversation.get_streaming_message_handler(item_id)
    assert existing_handler is not None, "Handler should exist after deltas"
    
    # This is what agents.py does when handler exists
    await existing_handler.update_message("Hello world!", replace_content=True)
    logger.info(f"After finalization, message count: {len(conversation.messages)}, messages: {[m.content for m in conversation.messages]}")
    
    # Verify still only 1 message in conversation.messages
    assert len(conversation.messages) == 1, f"Expected 1 message after finalization, got {len(conversation.messages)}: {[m.content for m in conversation.messages]}"
    assert conversation.messages[0].content == "Hello world!", f"Expected 'Hello world!', got '{conversation.messages[0].content}'"
    
    # Verify only 1 message in Stream Chat channel
    response = await channel.get_or_create(state=True, messages=MessagePaginationParams(limit=10))
    logger.info(f"Stream Chat messages: {[(m.id, m.text) for m in response.data.messages]}")
    assert len(response.data.messages) == 1, f"Expected 1 message in Stream, got {len(response.data.messages)}: {[(m.id, m.text) for m in response.data.messages]}"
    assert response.data.messages[0].text == "Hello world!", f"Expected 'Hello world!', got '{response.data.messages[0].text}'"
    
    logger.info(f"✅ Test passed: Only 1 message created")
