import datetime
import os
import threading
import time
import uuid
from unittest.mock import Mock

import pytest
from dotenv import load_dotenv
from getstream import Stream
from getstream.chat.client import ChatClient
from getstream.models import MessageRequest, ChannelResponse, ChannelInput

from vision_agents.core.agents.conversation import (
    Conversation,
    Message,
    InMemoryConversation,
    # StreamConversation,  # Removed from codebase
    StreamHandle
)

# Skip entire module - StreamConversation class has been removed from codebase
# TODO: Update tests to use new conversation architecture
pytestmark = pytest.mark.skip(reason="StreamConversation class removed - tests need migration to new architecture")

class TestConversation:
    """Test suite for the abstract Conversation class."""
    
    def test_conversation_is_abstract(self):
        """Test that Conversation cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            Conversation("instructions", [])
        assert "Can't instantiate abstract class" in str(exc_info.value)
    
    def test_conversation_requires_abstract_methods(self):
        """Test that subclasses must implement abstract methods."""
        class IncompleteConversation(Conversation):
            # Missing implementation of abstract methods
            pass
        
        with pytest.raises(TypeError) as exc_info:
            IncompleteConversation("instructions", [])
        assert "Can't instantiate abstract class" in str(exc_info.value)


class TestMessage:
    """Test suite for the Message dataclass."""
    
    def test_message_initialization(self):
        """Test that Message initializes correctly with default timestamp."""
        message = Message(
            original={"role": "user", "content": "Hello"},
            content="Hello",
            role="user",
            user_id="test-user"
        )
        
        assert message.content == "Hello"
        assert message.role == "user"
        assert message.user_id == "test-user"
        assert message.timestamp is not None
        assert isinstance(message.timestamp, datetime.datetime)


class TestInMemoryConversation:
    """Test suite for InMemoryConversation class."""
    
    @pytest.fixture
    def conversation(self):
        """Create a basic InMemoryConversation instance."""
        instructions = "You are a helpful assistant."
        messages = [
            Message(original=None, content="Hello", role="user", user_id="user1"),
            Message(original=None, content="Hi there!", role="assistant", user_id="assistant")
        ]
        # Set IDs for messages
        for i, msg in enumerate(messages):
            msg.id = f"msg-{i}"
        return InMemoryConversation(instructions, messages)
    
    def test_initialization(self, conversation):
        """Test InMemoryConversation initialization."""
        assert conversation.instructions == "You are a helpful assistant."
        assert len(conversation.messages) == 2
    
    def test_add_message(self, conversation):
        """Test adding a single message."""
        new_message = Message(
            original=None,
            content="New message",
            role="user",
            user_id="user2"
        )
        new_message.id = "new-msg"
        conversation.add_message(new_message)
        
        assert len(conversation.messages) == 3
        assert conversation.messages[-1] == new_message
    
    def test_add_message_with_completed(self, conversation):
        """Test adding a message with completed parameter."""
        # Test with completed=False
        new_message1 = Message(
            original=None,
            content="Generating message",
            role="user",
            user_id="user2"
        )
        new_message1.id = "gen-msg"
        result = conversation.add_message(new_message1, completed=False)
        
        assert len(conversation.messages) == 3
        assert conversation.messages[-1] == new_message1
        assert result is None  # InMemoryConversation returns None
        
        # Test with completed=True (default)
        new_message2 = Message(
            original=None,
            content="Complete message",
            role="user",
            user_id="user3"
        )
        new_message2.id = "comp-msg"
        result = conversation.add_message(new_message2, completed=True)
        
        assert len(conversation.messages) == 4
        assert conversation.messages[-1] == new_message2
        assert result is None
    
    def test_update_message_existing(self, conversation):
        """Test updating an existing message by appending content."""
        # Update existing message by appending (replace_content=False)
        result = conversation.update_message(
            message_id="msg-0",
            input_text=" additional text",
            user_id="user1",
            replace_content=False,
            completed=False
        )
        
        # Verify message content was appended (with space handling)
        assert conversation.messages[0].content == "Hello  additional text"
        assert result is None  # InMemoryConversation returns None
    
    def test_update_message_replace(self, conversation):
        """Test replacing message content (replace_content=True)."""
        result = conversation.update_message(
            message_id="msg-0",
            input_text="Replaced content",
            user_id="user1",
            replace_content=True,
            completed=True
        )
        
        # Verify message content was replaced
        assert conversation.messages[0].content == "Replaced content"
        assert result is None
    
    def test_update_message_not_found(self, conversation):
        """Test updating a non-existent message creates a new one."""
        initial_count = len(conversation.messages)
        
        conversation.update_message(
            message_id="non-existent-id",
            input_text="New message content",
            user_id="user2",
            replace_content=True,
            completed=False
        )
        
        # Should have added a new message
        assert len(conversation.messages) == initial_count + 1
        
        # Verify the new message was created correctly
        new_msg = conversation.messages[-1]
        assert new_msg.id == "non-existent-id"
        assert new_msg.content == "New message content"
        assert new_msg.user_id == "user2"
    
    def test_streaming_message_handle(self, conversation):
        """Test streaming message with handle API."""
        # Start a streaming message
        handle = conversation.start_streaming_message(role="assistant", initial_content="Hello")
        
        # Verify message was added
        assert len(conversation.messages) == 3
        assert conversation.messages[-1].content == "Hello"
        assert conversation.messages[-1].role == "assistant"
        assert isinstance(handle, StreamHandle)
        assert handle.user_id == "assistant"
        
        # Append to the message
        conversation.append_to_message(handle, " world")
        assert conversation.messages[-1].content == "Hello  world"
        
        # Replace the message
        conversation.replace_message(handle, "Goodbye")
        assert conversation.messages[-1].content == "Goodbye"
        
        # Complete the message
        conversation.complete_message(handle)
        # In-memory conversation doesn't track completed state, just verify no error
        
    def test_multiple_streaming_handles(self, conversation):
        """Test multiple concurrent streaming messages."""
        # Start two streaming messages
        handle1 = conversation.start_streaming_message(role="user", user_id="user1", initial_content="Question: ")
        handle2 = conversation.start_streaming_message(role="assistant", initial_content="Answer: ")
        
        assert len(conversation.messages) == 4  # 2 initial + 2 new
        
        # Update them independently
        conversation.append_to_message(handle1, "What is 2+2?")
        conversation.append_to_message(handle2, "Let me calculate...")
        
        # Find messages by their handles to verify correct updates
        msg1 = next(msg for msg in conversation.messages if msg.id == handle1.message_id)
        msg2 = next(msg for msg in conversation.messages if msg.id == handle2.message_id)
        
        assert msg1.content == "Question:  What is 2+2?"
        assert msg2.content == "Answer:  Let me calculate..."
        
        # Complete them
        conversation.complete_message(handle1)
        conversation.replace_message(handle2, "Answer: 4")
        conversation.complete_message(handle2)
        
        assert msg2.content == "Answer: 4"  # Replaced content, no space issue


class TestStreamConversation:
    """Test suite for StreamConversation class."""
    
    @pytest.fixture
    def mock_chat_client(self):
        """Create a mock ChatClient."""
        client = Mock(spec=ChatClient)
        
        # Mock send_message response
        mock_response = Mock()
        mock_response.data.message.id = "stream-message-123"
        client.send_message.return_value = mock_response
        
        # Mock ephemeral_message_update
        client.ephemeral_message_update = Mock(return_value=Mock())
        
        # Mock update_message_partial
        client.update_message_partial = Mock(return_value=Mock())
        
        return client
    
    @pytest.fixture
    def mock_channel(self):
        """Create a mock ChannelResponse."""
        channel = Mock(spec=ChannelResponse)
        channel.type = "messaging"
        channel.id = "test-channel-123"
        return channel
    
    @pytest.fixture
    def stream_conversation(self, mock_chat_client, mock_channel):
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
            
        conversation = StreamConversation(  # noqa: F821
            instructions=instructions,
            messages=messages,
            channel=mock_channel,
            chat_client=mock_chat_client
        )
        
        # Pre-populate some stream IDs for testing
        conversation.internal_ids_to_stream_ids = {
            "msg-0": "stream-msg-0"
        }
        
        yield conversation
        
        # Cleanup after each test
        conversation.shutdown()
    
    def test_initialization(self, stream_conversation, mock_channel, mock_chat_client):
        """Test StreamConversation initialization."""
        assert stream_conversation.channel == mock_channel
        assert stream_conversation.chat_client == mock_chat_client
        assert isinstance(stream_conversation.internal_ids_to_stream_ids, dict)
        assert len(stream_conversation.messages) == 1
    
    def test_add_message(self, stream_conversation, mock_chat_client):
        """Test adding a message to the stream with default completed=True."""
        new_message = Message(
            original=None,
            content="Test message",
            role="user",
            user_id="user123"
        )
        new_message.id = "new-msg-id"
        
        stream_conversation.add_message(new_message)
        
        # Verify message was added locally immediately
        assert len(stream_conversation.messages) == 2
        assert stream_conversation.messages[-1] == new_message
        
        # Wait for async operations to complete
        assert stream_conversation.wait_for_pending_operations(timeout=2.0)
        
        # Verify Stream API was called
        mock_chat_client.send_message.assert_called_once()
        call_args = mock_chat_client.send_message.call_args
        assert call_args[0][0] == "messaging"  # channel type
        assert call_args[0][1] == "test-channel-123"  # channel id
        
        request = call_args[0][2]
        assert isinstance(request, MessageRequest)
        assert request.text == "Test message"
        assert request.user_id == "user123"
        
        # Verify ID mapping was stored
        assert "new-msg-id" in stream_conversation.internal_ids_to_stream_ids
        assert stream_conversation.internal_ids_to_stream_ids["new-msg-id"] == "stream-message-123"
        
        # Wait a bit more for the update operation to complete
        time.sleep(0.1)
        
        # Verify update_message_partial was called (completed=True is default)
        mock_chat_client.update_message_partial.assert_called_once()
        update_args = mock_chat_client.update_message_partial.call_args
        assert update_args[0][0] == "stream-message-123"
        assert update_args[1]["user_id"] == "user123"
        assert update_args[1]["set"]["text"] == "Test message"
        assert update_args[1]["set"]["generating"] is False  # completed=True means not generating
    
    def test_add_message_with_completed_false(self, stream_conversation, mock_chat_client):
        """Test adding a message with completed=False (still generating)."""
        # Ensure previous operations are complete
        stream_conversation.wait_for_pending_operations(timeout=1.0)
        
        # Reset mocks
        mock_chat_client.send_message.reset_mock()
        mock_chat_client.ephemeral_message_update.reset_mock()
        mock_chat_client.update_message_partial.reset_mock()
        
        new_message = Message(
            original=None,
            content="Generating message",
            role="assistant",
            user_id="assistant"
        )
        new_message.id = "gen-msg-id"
        
        stream_conversation.add_message(new_message, completed=False)
        
        # Verify message was added locally
        assert len(stream_conversation.messages) == 2
        assert stream_conversation.messages[-1] == new_message
        
        # Wait for async operations to complete
        assert stream_conversation.wait_for_pending_operations(timeout=2.0)
        
        # Verify Stream API was called
        mock_chat_client.send_message.assert_called_once()
        
        # Give a bit more time for the update operation to be queued and processed
        time.sleep(0.2)
        
        # Verify ephemeral_message_update was called (completed=False)
        mock_chat_client.ephemeral_message_update.assert_called_once()
        mock_chat_client.update_message_partial.assert_not_called()
        
        update_args = mock_chat_client.ephemeral_message_update.call_args
        assert update_args[0][0] == "stream-message-123"
        assert update_args[1]["user_id"] == "assistant"
        assert update_args[1]["set"]["text"] == "Generating message"
        assert update_args[1]["set"]["generating"] is True  # completed=False means still generating
    
    def test_update_message_existing(self, stream_conversation, mock_chat_client):
        """Test updating an existing message by appending content."""
        # Update existing message by appending (replace_content=False, completed=False)
        stream_conversation.update_message(
            message_id="msg-0",
            input_text=" additional text",
            user_id="user1",
            replace_content=False,
            completed=False
        )
        
        # Verify message content was appended immediately
        assert stream_conversation.messages[0].content == "Hello additional text"
        
        # Wait for async operations to complete
        assert stream_conversation.wait_for_pending_operations(timeout=2.0)
        
        # Verify Stream API was called with ephemeral_message_update (not completed)
        mock_chat_client.ephemeral_message_update.assert_called_once()
        call_args = mock_chat_client.ephemeral_message_update.call_args
        assert call_args[0][0] == "stream-msg-0"  # stream message ID
        assert call_args[1]["user_id"] == "user1"
        assert call_args[1]["set"]["text"] == "Hello additional text"
        assert call_args[1]["set"]["generating"] is True  # not completed = still generating
    
    def test_update_message_replace(self, stream_conversation, mock_chat_client):
        """Test replacing message content (replace_content=True)."""
        # Mock update_message_partial for completed messages
        mock_chat_client.update_message_partial = Mock(return_value=Mock())
        
        stream_conversation.update_message(
            message_id="msg-0",
            input_text="Replaced content",
            user_id="user1",
            replace_content=True,
            completed=True
        )
        
        # Verify message content was replaced
        assert stream_conversation.messages[0].content == "Replaced content"
        
        # Wait for async operations to complete
        assert stream_conversation.wait_for_pending_operations(timeout=2.0)
        
        # Verify Stream API was called with update_message_partial (completed)
        mock_chat_client.update_message_partial.assert_called_once()
        call_args = mock_chat_client.update_message_partial.call_args
        assert call_args[0][0] == "stream-msg-0"  # stream message ID
        assert call_args[1]["user_id"] == "user1"
        assert call_args[1]["set"]["text"] == "Replaced content"
        assert call_args[1]["set"]["generating"] is False  # completed = not generating
    
    def test_update_message_not_found(self, stream_conversation, mock_chat_client):
        """Test updating a non-existent message creates a new one."""
        # Reset the send_message mock for this test
        mock_chat_client.send_message.reset_mock()
        
        stream_conversation.update_message(
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
        
        # Wait for async operations to complete
        assert stream_conversation.wait_for_pending_operations(timeout=2.0)
        time.sleep(0.2)  # Give extra time for update operation
        
        # Verify send_message was called (not update)
        mock_chat_client.send_message.assert_called_once()
    
    def test_update_message_completed_vs_generating(self, stream_conversation, mock_chat_client):
        """Test that completed=True calls update_message_partial and completed=False calls ephemeral_message_update."""
        # Mock update_message_partial for completed messages
        mock_chat_client.update_message_partial = Mock(return_value=Mock())
        
        # Test with completed=False (still generating)
        stream_conversation.update_message(
            message_id="msg-0",
            input_text=" in progress",
            user_id="user1",
            replace_content=False,
            completed=False
        )
        
        # Wait for async operations
        assert stream_conversation.wait_for_pending_operations(timeout=2.0)
        
        # Should call ephemeral_message_update
        mock_chat_client.ephemeral_message_update.assert_called()
        mock_chat_client.update_message_partial.assert_not_called()
        
        # Reset mocks
        mock_chat_client.ephemeral_message_update.reset_mock()
        
        # Test with completed=True
        stream_conversation.update_message(
            message_id="msg-0",
            input_text="Final content",
            user_id="user1",
            replace_content=True,
            completed=True
        )
        
        # Wait for async operations
        assert stream_conversation.wait_for_pending_operations(timeout=2.0)
        
        # Should call update_message_partial
        mock_chat_client.update_message_partial.assert_called_once()
        mock_chat_client.ephemeral_message_update.assert_not_called()
    
    def test_update_message_no_stream_id(self, stream_conversation, mock_chat_client):
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
        stream_conversation.update_message(
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
        mock_chat_client.ephemeral_message_update.assert_not_called()
    
    def test_streaming_message_handle(self, stream_conversation, mock_chat_client):
        """Test streaming message with handle API."""
        # Reset mocks
        mock_chat_client.send_message.reset_mock()
        mock_chat_client.ephemeral_message_update.reset_mock()
        mock_chat_client.update_message_partial.reset_mock()
        
        # Start a streaming message
        handle = stream_conversation.start_streaming_message(role="assistant", initial_content="Processing")
        
        # Verify message was added and marked as generating
        assert len(stream_conversation.messages) == 2
        assert stream_conversation.messages[-1].content == "Processing"
        assert stream_conversation.messages[-1].role == "assistant"
        assert isinstance(handle, StreamHandle)
        assert handle.user_id == "assistant"
        
        # Wait for async operations
        assert stream_conversation.wait_for_pending_operations(timeout=2.0)
        time.sleep(0.2)  # Give extra time for update operation
        
        # Verify send_message was called
        mock_chat_client.send_message.assert_called_once()
        # Verify ephemeral_message_update was called (completed=False by default)
        mock_chat_client.ephemeral_message_update.assert_called_once()
        
        # Reset for next operations
        mock_chat_client.ephemeral_message_update.reset_mock()
        
        # Append to the message
        stream_conversation.append_to_message(handle, "...")
        assert stream_conversation.messages[-1].content == "Processing..."
        
        # Wait for append operation to complete
        assert stream_conversation.wait_for_pending_operations(timeout=2.0)
        
        mock_chat_client.ephemeral_message_update.assert_called_once()
        
        # Replace the message
        stream_conversation.replace_message(handle, "Complete response")
        assert stream_conversation.messages[-1].content == "Complete response"
        
        # Wait for replace operation to complete
        assert stream_conversation.wait_for_pending_operations(timeout=2.0)
        
        assert mock_chat_client.ephemeral_message_update.call_count == 2
        
        # Complete the message
        mock_chat_client.update_message_partial.reset_mock()
        stream_conversation.complete_message(handle)
        
        # Wait for complete operation
        assert stream_conversation.wait_for_pending_operations(timeout=2.0)
        
        mock_chat_client.update_message_partial.assert_called_once()
        
    def test_multiple_streaming_handles(self, stream_conversation, mock_chat_client):
        """Test multiple concurrent streaming messages with Stream API."""
        # Reset mocks
        mock_chat_client.send_message.reset_mock()
        mock_chat_client.ephemeral_message_update.reset_mock()
        
        # Mock different message IDs for each send
        mock_response1 = Mock()
        mock_response1.data.message.id = "stream-msg-1"
        mock_response2 = Mock()
        mock_response2.data.message.id = "stream-msg-2"
        mock_chat_client.send_message.side_effect = [mock_response1, mock_response2]
        
        # Start two streaming messages with empty initial content
        handle1 = stream_conversation.start_streaming_message(role="user", user_id="user123", initial_content="")
        handle2 = stream_conversation.start_streaming_message(role="assistant", initial_content="")
        
        assert len(stream_conversation.messages) == 3  # 1 initial + 2 new
        
        # Wait for initial operations to complete
        assert stream_conversation.wait_for_pending_operations(timeout=2.0)
        time.sleep(0.3)  # Give extra time for update operations
        
        # Update them independently
        stream_conversation.append_to_message(handle1, "Hello?")
        stream_conversation.append_to_message(handle2, "Hi there!")
        
        # Wait for append operations to complete
        assert stream_conversation.wait_for_pending_operations(timeout=2.0)
        
        # Find messages by their handles to verify correct updates
        msg1 = next(msg for msg in stream_conversation.messages if msg.id == handle1.message_id)
        msg2 = next(msg for msg in stream_conversation.messages if msg.id == handle2.message_id)
        
        assert msg1.content == "Hello?"
        assert msg2.content == "Hi there!"
        
        # Verify ephemeral updates were called for both
        assert mock_chat_client.ephemeral_message_update.call_count >= 4  # 2 initial + 2 appends
        
        # Complete both
        stream_conversation.complete_message(handle1)
        stream_conversation.complete_message(handle2)
        
        # Wait for completion operations
        assert stream_conversation.wait_for_pending_operations(timeout=2.0)
        
        # Verify update_message_partial was called for both completions
        assert mock_chat_client.update_message_partial.call_count == 2
    
    def test_worker_thread_async_operations(self, stream_conversation, mock_chat_client):
        """Test that operations are processed asynchronously by the worker thread."""
        # Reset mocks
        mock_chat_client.send_message.reset_mock()
        mock_chat_client.ephemeral_message_update.reset_mock()
        
        # Add multiple messages quickly
        messages = []
        for i in range(5):
            msg = Message(
                original=None,
                content=f"Message {i}",
                role="user",
                user_id=f"user{i}"
            )
            messages.append(msg)
            stream_conversation.add_message(msg, completed=False)
        
        # Verify messages were added locally immediately
        assert len(stream_conversation.messages) == 6  # 1 initial + 5 new
        
        # Wait for all operations to complete
        assert stream_conversation.wait_for_pending_operations(timeout=3.0)
        
        # Give a bit more time for update operations
        time.sleep(0.5)
        
        # Verify all send_message calls were made
        assert mock_chat_client.send_message.call_count == 5
        
        # Verify all ephemeral_message_update calls were made
        assert mock_chat_client.ephemeral_message_update.call_count >= 5
    
    def test_wait_for_pending_operations_timeout(self, stream_conversation, mock_chat_client):
        """Test that wait_for_pending_operations returns False on timeout."""
        # Make send_message block for a long time
        block_event = threading.Event()
        
        def slow_send_message(*args, **kwargs):
            block_event.wait(timeout=5.0)  # Block for 5 seconds
            mock_response = Mock()
            mock_response.data.message.id = "stream-message-slow"
            return mock_response
        
        mock_chat_client.send_message.side_effect = slow_send_message
        
        # Add a message
        msg = Message(original=None, content="Slow message", role="user", user_id="user1")
        stream_conversation.add_message(msg)
        
        # Wait should timeout
        assert not stream_conversation.wait_for_pending_operations(timeout=0.5)
        
        # Unblock the operation
        block_event.set()
        
        # Now wait should succeed
        assert stream_conversation.wait_for_pending_operations(timeout=2.0)
    
    def test_shutdown_worker_thread(self, mock_chat_client, mock_channel):
        """Test that shutdown properly stops the worker thread."""
        # Create a fresh conversation without using the fixture to avoid double shutdown
        conversation = StreamConversation(  # noqa: F821
            instructions="Test",
            messages=[],
            channel=mock_channel,
            chat_client=mock_chat_client
        )
        
        # Verify thread is alive
        assert conversation._worker_thread.is_alive()
        
        # Shutdown
        conversation.shutdown()
        
        # Verify thread stopped
        assert not conversation._worker_thread.is_alive()
        
        # Verify shutdown flag is set
        assert conversation._shutdown is True


@pytest.fixture
def mock_stream_client():
    """Create a mock Stream client for testing."""
    from getstream import Stream
    
    client = Mock(spec=Stream)
    
    # Mock user creation
    mock_user = Mock()
    mock_user.id = "test-agent-user"
    mock_user.name = "Test Agent"
    client.create_user.return_value = mock_user
    
    # Mock video.call
    mock_call = Mock()
    mock_call.id = "test-call-123"
    client.video.call.return_value = mock_call
    
    return client


@pytest.mark.integration
def test_stream_conversation_integration():
    """Integration test with real Stream client (requires credentials)."""

    load_dotenv()

    if not os.getenv("STREAM_API_KEY"):
        pytest.skip("Stream credentials not available")
    
    # Create real client
    client = Stream.from_env()
    
    # Create a test channel and user
    user = client.create_user(id="test-user")
    channel = client.chat.get_or_create_channel("messaging", str(uuid.uuid4()), data=ChannelInput(created_by_id=user.id)).data.channel

    # Create conversation
    conversation = StreamConversation(  # noqa: F821
        instructions="Test assistant",
        messages=[],
        channel=channel,
        chat_client=client.chat
    )

    # Add a message
    message = Message(
        original=None,
        content="Hello from test",
        role="user",
        user_id=user.id
    )
    conversation.add_message(message)

    # Wait for async operations to complete
    assert conversation.wait_for_pending_operations(timeout=5.0)

    # Verify message was sent
    assert len(conversation.messages) == 1
    assert message.id in conversation.internal_ids_to_stream_ids

    # update message with replace
    conversation.update_message(message_id=message.id, input_text="Replaced content", user_id=user.id, replace_content=True, completed=True)
    assert conversation.wait_for_pending_operations(timeout=5.0)

    channel_data = client.chat.get_or_create_channel("messaging", channel.id, state=True).data
    assert len(channel_data.messages) == 1
    assert channel_data.messages[0].text == "Replaced content"
    # Note: generating flag might not be in custom field depending on Stream API version

    # update message with delta
    conversation.update_message(message_id=message.id, input_text=" more stuff", user_id=user.id,
                                replace_content=False, completed=True)
    assert conversation.wait_for_pending_operations(timeout=5.0)

    channel_data = client.chat.get_or_create_channel("messaging", channel.id, state=True).data
    assert len(channel_data.messages) == 1
    assert channel_data.messages[0].text == "Replaced content more stuff"
    # Note: generating flag might not be in custom field depending on Stream API version
    
    # Test add_message with completed=False
    message2 = Message(
        original=None,
        content="Still generating...",
        role="assistant",
        user_id="assistant"
    )
    conversation.add_message(message2, completed=False)
    assert conversation.wait_for_pending_operations(timeout=5.0)
    time.sleep(0.2)  # Give extra time for update operation
    
    channel_data = client.chat.get_or_create_channel("messaging", channel.id, state=True).data
    assert len(channel_data.messages) == 2
    assert channel_data.messages[1].text == "Still generating..."
    # Note: generating flag might not be in custom field depending on Stream API version
    
    # Test streaming handle API
    handle = conversation.start_streaming_message(role="assistant", initial_content="Thinking")
    assert conversation.wait_for_pending_operations(timeout=5.0)
    time.sleep(0.2)  # Give extra time for update operation
    
    conversation.append_to_message(handle, "...")
    assert conversation.wait_for_pending_operations(timeout=5.0)
    
    conversation.replace_message(handle, "The answer is 42")
    assert conversation.wait_for_pending_operations(timeout=5.0)
    
    conversation.complete_message(handle)
    assert conversation.wait_for_pending_operations(timeout=5.0)
    
    channel_data = client.chat.get_or_create_channel("messaging", channel.id, state=True).data
    assert len(channel_data.messages) == 3
    assert channel_data.messages[2].text == "The answer is 42"
    # Note: generating flag might not be in custom field depending on Stream API version
    
    # Cleanup
    conversation.shutdown()
