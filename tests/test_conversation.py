import datetime
import uuid

import pytest
from unittest.mock import Mock, AsyncMock

from getstream.chat.client import ChatClient
from getstream.models import MessageRequest, ChannelResponse, ChannelInput

from stream_agents.core.agents.conversation import (
    Conversation,
    Message,
    InMemoryConversation,
    StreamConversation,
    StreamHandle
)
from getstream import AsyncStream
from dotenv import load_dotenv

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
    
    async def test_add_message(self, conversation):
        """Test adding a single message."""
        new_message = Message(
            original=None,
            content="New message",
            role="user",
            user_id="user2"
        )
        new_message.id = "new-msg"
        await conversation.add_message(new_message)
        
        assert len(conversation.messages) == 3
        assert conversation.messages[-1] == new_message
    
    async def test_add_message_with_completed(self, conversation):
        """Test adding a message with completed parameter."""
        # Test with completed=False
        new_message1 = Message(
            original=None,
            content="Generating message",
            role="user",
            user_id="user2"
        )
        new_message1.id = "gen-msg"
        result = await conversation.add_message(new_message1, completed=False)
        
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
        result = await conversation.add_message(new_message2, completed=True)
        
        assert len(conversation.messages) == 4
        assert conversation.messages[-1] == new_message2
        assert result is None
    
    async def test_update_message_existing(self, conversation):
        """Test updating an existing message by appending content."""
        # Update existing message by appending (replace_content=False)
        result = await conversation.update_message(
            message_id="msg-0",
            input_text=" additional text",
            user_id="user1",
            replace_content=False,
            completed=False
        )
        
        # Verify message content was appended (with space handling)
        assert conversation.messages[0].content == "Hello additional text"
        assert result is None  # InMemoryConversation returns None
    
    async def test_update_message_replace(self, conversation):
        """Test replacing message content (replace_content=True)."""
        result = await conversation.update_message(
            message_id="msg-0",
            input_text="Replaced content",
            user_id="user1",
            replace_content=True,
            completed=True
        )
        
        # Verify message content was replaced
        assert conversation.messages[0].content == "Replaced content"
        assert result is None
    
    async def test_update_message_not_found(self, conversation):
        """Test updating a non-existent message creates a new one."""
        initial_count = len(conversation.messages)
        
        result = await conversation.update_message(
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
    
    async def test_streaming_message_handle(self, conversation):
        """Test streaming message with handle API."""
        # Start a streaming message
        handle = await conversation.start_streaming_message(role="assistant", initial_content="Hello")
        
        # Verify message was added
        assert len(conversation.messages) == 3
        assert conversation.messages[-1].content == "Hello"
        assert conversation.messages[-1].role == "assistant"
        assert isinstance(handle, StreamHandle)
        assert handle.user_id == "assistant"
        
        # Append to the message
        await conversation.append_to_message(handle, " world")
        assert conversation.messages[-1].content == "Hello world"
        
        # Replace the message
        await conversation.replace_message(handle, "Goodbye")
        assert conversation.messages[-1].content == "Goodbye"
        
        # Complete the message
        await conversation.complete_message(handle)
        # In-memory conversation doesn't track completed state, just verify no error
        
    async def test_multiple_streaming_handles(self, conversation):
        """Test multiple concurrent streaming messages."""
        # Start two streaming messages
        handle1 = await conversation.start_streaming_message(role="user", user_id="user1", initial_content="Question: ")
        handle2 = await conversation.start_streaming_message(role="assistant", initial_content="Answer: ")
        
        assert len(conversation.messages) == 4  # 2 initial + 2 new
        
        # Update them independently
        await conversation.append_to_message(handle1, "What is 2+2?")
        await conversation.append_to_message(handle2, "Let me calculate...")
        
        # Find messages by their handles to verify correct updates
        msg1 = next(msg for msg in conversation.messages if msg.id == handle1.message_id)
        msg2 = next(msg for msg in conversation.messages if msg.id == handle2.message_id)
        
        assert msg1.content == "Question: What is 2+2?"
        assert msg2.content == "Answer: Let me calculate..."
        
        # Complete them
        await conversation.complete_message(handle1)
        await conversation.replace_message(handle2, "Answer: 4")
        await conversation.complete_message(handle2)
        
        assert msg2.content == "Answer: 4"


class TestStreamConversation:
    """Test suite for StreamConversation class."""
    
    @pytest.fixture
    def mock_chat_client(self):
        """Create a mock ChatClient."""
        client = Mock(spec=ChatClient)
        
        # Mock send_message response
        mock_response = Mock()
        mock_response.data.message.id = "stream-message-123"
        client.send_message = AsyncMock(return_value=mock_response)
        
        # Mock ephemeral_message_update
        client.ephemeral_message_update = AsyncMock(return_value=Mock())
        
        # Mock update_message_partial
        client.update_message_partial = AsyncMock(return_value=Mock())
        
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
            
        conversation = StreamConversation(
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
    
    def test_initialization(self, stream_conversation, mock_channel, mock_chat_client):
        """Test StreamConversation initialization."""
        assert stream_conversation.channel == mock_channel
        assert stream_conversation.chat_client == mock_chat_client
        assert isinstance(stream_conversation.internal_ids_to_stream_ids, dict)
        assert len(stream_conversation.messages) == 1
    
    async def test_add_message(self, stream_conversation, mock_chat_client):
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
        
        # Verify update_message_partial was called (completed=True is default)
        mock_chat_client.update_message_partial.assert_called_once()
        update_args = mock_chat_client.update_message_partial.call_args
        assert update_args[0][0] == "stream-message-123"
        assert update_args[1]["user_id"] == "user123"
        assert update_args[1]["set"]["text"] == "Test message"
        assert update_args[1]["set"]["generating"] is False  # completed=True means not generating
    
    async def test_add_message_with_completed_false(self, stream_conversation, mock_chat_client):
        """Test adding a message with completed=False (still generating)."""
        
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
        
        await stream_conversation.add_message(new_message, completed=False)
        
        # Verify message was added locally
        assert len(stream_conversation.messages) == 2
        assert stream_conversation.messages[-1] == new_message
        
        # Verify Stream API was called
        mock_chat_client.send_message.assert_called_once()
        
        # Verify ephemeral_message_update was called (completed=False)
        mock_chat_client.ephemeral_message_update.assert_called_once()
        mock_chat_client.update_message_partial.assert_not_called()
        
        update_args = mock_chat_client.ephemeral_message_update.call_args
        assert update_args[0][0] == "stream-message-123"
        assert update_args[1]["user_id"] == "assistant"
        assert update_args[1]["set"]["text"] == "Generating message"
        assert update_args[1]["set"]["generating"] is True  # completed=False means still generating
    
    async def test_update_message_existing(self, stream_conversation, mock_chat_client):
        """Test updating an existing message by appending content."""
        # Update existing message by appending (replace_content=False, completed=False)
        result = await stream_conversation.update_message(
            message_id="msg-0",
            input_text=" additional text",
            user_id="user1",
            replace_content=False,
            completed=False
        )
        
        # Verify message content was appended immediately
        assert stream_conversation.messages[0].content == "Hello additional text"
        
        # Verify Stream API was called with ephemeral_message_update (not completed)
        mock_chat_client.ephemeral_message_update.assert_called_once()
        call_args = mock_chat_client.ephemeral_message_update.call_args
        assert call_args[0][0] == "stream-msg-0"  # stream message ID
        assert call_args[1]["user_id"] == "user1"
        assert call_args[1]["set"]["text"] == "Hello additional text"
        assert call_args[1]["set"]["generating"] is True  # not completed = still generating
    
    async def test_update_message_replace(self, stream_conversation, mock_chat_client):
        """Test replacing message content (replace_content=True)."""
        # Mock update_message_partial for completed messages
        mock_chat_client.update_message_partial = AsyncMock(return_value=Mock())
        
        result = await stream_conversation.update_message(
            message_id="msg-0",
            input_text="Replaced content",
            user_id="user1",
            replace_content=True,
            completed=True
        )
        
        # Verify message content was replaced
        assert stream_conversation.messages[0].content == "Replaced content"
        
        # Verify Stream API was called with update_message_partial (completed)
        mock_chat_client.update_message_partial.assert_called_once()
        call_args = mock_chat_client.update_message_partial.call_args
        assert call_args[0][0] == "stream-msg-0"  # stream message ID
        assert call_args[1]["user_id"] == "user1"
        assert call_args[1]["set"]["text"] == "Replaced content"
        assert call_args[1]["set"]["generating"] is False  # completed = not generating
    
    async def test_update_message_not_found(self, stream_conversation, mock_chat_client):
        """Test updating a non-existent message creates a new one."""
        # Reset the send_message mock for this test
        mock_chat_client.send_message.reset_mock()
        
        result = await stream_conversation.update_message(
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
        mock_chat_client.send_message.assert_called_once()
    
    async def test_update_message_completed_vs_generating(self, stream_conversation, mock_chat_client):
        """Test that completed=True calls update_message_partial and completed=False calls ephemeral_message_update."""
        # Mock update_message_partial for completed messages
        mock_chat_client.update_message_partial = AsyncMock(return_value=Mock())
        
        # Test with completed=False (still generating)
        await stream_conversation.update_message(
            message_id="msg-0",
            input_text=" in progress",
            user_id="user1",
            replace_content=False,
            completed=False
        )
        
        # Should call ephemeral_message_update
        mock_chat_client.ephemeral_message_update.assert_called()
        mock_chat_client.update_message_partial.assert_not_called()
        
        # Reset mocks
        mock_chat_client.ephemeral_message_update.reset_mock()
        
        # Test with completed=True
        await stream_conversation.update_message(
            message_id="msg-0",
            input_text="Final content",
            user_id="user1",
            replace_content=True,
            completed=True
        )
        
        # Should call update_message_partial
        mock_chat_client.update_message_partial.assert_called_once()
        mock_chat_client.ephemeral_message_update.assert_not_called()
    
    async def test_update_message_no_stream_id(self, stream_conversation, mock_chat_client):
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
        result = await stream_conversation.update_message(
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
    
    async def test_streaming_message_handle(self, stream_conversation, mock_chat_client):
        """Test streaming message with handle API."""
        # Reset mocks
        mock_chat_client.send_message.reset_mock()
        mock_chat_client.ephemeral_message_update.reset_mock()
        mock_chat_client.update_message_partial.reset_mock()
        
        # Start a streaming message
        handle = await stream_conversation.start_streaming_message(role="assistant", initial_content="Processing")
        
        # Verify message was added and marked as generating
        assert len(stream_conversation.messages) == 2
        assert stream_conversation.messages[-1].content == "Processing"
        assert stream_conversation.messages[-1].role == "assistant"
        assert isinstance(handle, StreamHandle)
        assert handle.user_id == "assistant"
        
        
        # Verify send_message was called
        mock_chat_client.send_message.assert_called_once()
        # Verify ephemeral_message_update was called (completed=False by default)
        mock_chat_client.ephemeral_message_update.assert_called_once()
        
        # Reset for next operations
        mock_chat_client.ephemeral_message_update.reset_mock()
        
        # Append to the message
        await stream_conversation.append_to_message(handle, "...")
        assert stream_conversation.messages[-1].content == "Processing..."
        
        mock_chat_client.ephemeral_message_update.assert_called_once()
        
        # Replace the message
        await stream_conversation.replace_message(handle, "Complete response")
        assert stream_conversation.messages[-1].content == "Complete response"
        
        assert mock_chat_client.ephemeral_message_update.call_count == 2
        
        # Complete the message
        mock_chat_client.update_message_partial.reset_mock()
        await stream_conversation.complete_message(handle)
        
        mock_chat_client.update_message_partial.assert_called_once()
        
    async def test_multiple_streaming_handles(self, stream_conversation, mock_chat_client):
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
        assert mock_chat_client.ephemeral_message_update.call_count >= 4  # 2 initial + 2 appends
        
        # Complete both
        await stream_conversation.complete_message(handle1)
        await stream_conversation.complete_message(handle2)
        
        # Verify update_message_partial was called for both completions
        assert mock_chat_client.update_message_partial.call_count == 2
    
    async def test_async_operations(self, stream_conversation, mock_chat_client):
        """Test that operations are processed asynchronously."""
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
            await stream_conversation.add_message(msg, completed=False)
        
        # Verify messages were added locally immediately
        assert len(stream_conversation.messages) == 6  # 1 initial + 5 new
        
        # Verify all send_message calls were made
        assert mock_chat_client.send_message.call_count == 5
        
        # Verify all ephemeral_message_update calls were made
        assert mock_chat_client.ephemeral_message_update.call_count >= 5
    
    async def test_concurrent_streaming_handles(self, stream_conversation, mock_chat_client):
        """Test concurrent streaming operations don't interfere with each other."""
        import asyncio
        
        # Reset mocks
        mock_chat_client.send_message.reset_mock()
        mock_chat_client.ephemeral_message_update.reset_mock()
        mock_chat_client.update_message_partial.reset_mock()
        
        # Mock unique message IDs for each send
        message_ids = [f"stream-msg-{i}" for i in range(20)]
        mock_responses = []
        for msg_id in message_ids:
            mock_resp = Mock()
            mock_resp.data.message.id = msg_id
            mock_responses.append(mock_resp)
        
        mock_chat_client.send_message.side_effect = mock_responses
        
        # Test 1: Create multiple streaming messages concurrently
        async def create_and_update_message(msg_num: int):
            # Create a streaming message
            handle = await stream_conversation.start_streaming_message(
                role="user" if msg_num % 2 == 0 else "assistant",
                user_id=f"user{msg_num}",
                initial_content=f"Msg{msg_num}: "
            )
            
            # Update it multiple times
            for i in range(3):
                await stream_conversation.append_to_message(handle, f"p{i} ")
            
            # Complete it
            await stream_conversation.complete_message(handle)
            
            return handle
        
        # Run concurrent operations
        handles = await asyncio.gather(*[
            create_and_update_message(i) for i in range(5)
        ])
        
        # Verify all handles have unique message IDs
        handle_ids = [h.message_id for h in handles]
        assert len(set(handle_ids)) == 5
        
        # Verify messages have correct content
        for i, handle in enumerate(handles):
            msg = next(m for m in stream_conversation.messages if m.id == handle.message_id)
            assert msg.content == f"Msg{i}: p0 p1 p2 "
        
        # Test 2: Concurrent replacements don't interfere
        async def replace_message_content(handle: StreamHandle, new_content: str):
            await stream_conversation.replace_message(handle, new_content)
            return handle
        
        # Replace all messages concurrently with different content
        await asyncio.gather(*[
            replace_message_content(handle, f"Replaced{i}")
            for i, handle in enumerate(handles)
        ])
        
        # Verify each message has its own unique replaced content
        for i, handle in enumerate(handles):
            msg = next(m for m in stream_conversation.messages if m.id == handle.message_id)
            assert msg.content == f"Replaced{i}"
        
        # Test 3: Rapid fire operations
        rapid_handles = []
        
        async def rapid_operations():
            local_handles = []
            # Create 3 messages as fast as possible
            for i in range(3):
                h = await stream_conversation.start_streaming_message(
                    role="assistant",
                    initial_content=f"Rapid{i}: "
                )
                local_handles.append(h)
            
            # Update them all concurrently
            update_tasks = []
            for h in local_handles:
                for j in range(5):
                    update_tasks.append(
                        stream_conversation.append_to_message(h, f"[{j}]")
                    )
            await asyncio.gather(*update_tasks)
            
            return local_handles
        
        # Run multiple rapid operations concurrently
        rapid_results = await asyncio.gather(*[rapid_operations() for _ in range(2)])
        
        for handles_list in rapid_results:
            rapid_handles.extend(handles_list)
        
        # Verify all rapid messages have unique IDs
        rapid_ids = [h.message_id for h in rapid_handles]
        assert len(set(rapid_ids)) == len(rapid_ids)
        
        # Each rapid message should have all its updates
        for h in rapid_handles:
            msg = next(m for m in stream_conversation.messages if m.id == h.message_id)
            assert "[0][1][2][3][4]" in msg.content
        
        # Final verification: internal mapping integrity
        assert len(stream_conversation.internal_ids_to_stream_ids) >= len(handles) + len(rapid_handles)
    
    async def test_handle_isolation(self, stream_conversation, mock_chat_client):
        """Test that handles remain isolated and don't cross-contaminate."""
        import asyncio
        
        # Reset mocks
        mock_chat_client.send_message.reset_mock()
        mock_chat_client.ephemeral_message_update.reset_mock()
        
        # Mock unique message IDs
        mock_responses = []
        for i in range(10):
            mock_resp = Mock()
            mock_resp.data.message.id = f"isolated-msg-{i}"
            mock_responses.append(mock_resp)
        
        mock_chat_client.send_message.side_effect = mock_responses
        
        # Create handles with specific patterns
        handle1 = await stream_conversation.start_streaming_message(
            role="user",
            user_id="user1",
            initial_content="HANDLE1:"
        )
        
        handle2 = await stream_conversation.start_streaming_message(
            role="assistant",
            user_id="assistant1",
            initial_content="HANDLE2:"
        )
        
        # Concurrent updates with distinct patterns
        async def update_handle1():
            for i in range(5):
                await stream_conversation.append_to_message(handle1, f"A{i}")
                await asyncio.sleep(0.001)  # Small delay
        
        async def update_handle2():
            for i in range(5):
                await stream_conversation.append_to_message(handle2, f"B{i}")
                await asyncio.sleep(0.001)  # Small delay
        
        # Run updates concurrently
        await asyncio.gather(update_handle1(), update_handle2())
        
        # Verify messages didn't cross-contaminate
        msg1 = next(m for m in stream_conversation.messages if m.id == handle1.message_id)
        msg2 = next(m for m in stream_conversation.messages if m.id == handle2.message_id)
        
        # Check handle1 only has A updates
        assert "HANDLE1:" in msg1.content
        assert all(f"A{i}" in msg1.content for i in range(5))
        assert not any(f"B{i}" in msg1.content for i in range(5))
        assert "HANDLE2:" not in msg1.content
        
        # Check handle2 only has B updates
        assert "HANDLE2:" in msg2.content
        assert all(f"B{i}" in msg2.content for i in range(5))
        assert not any(f"A{i}" in msg2.content for i in range(5))
        assert "HANDLE1:" not in msg2.content
        
        # Test concurrent replacements maintain isolation
        await asyncio.gather(
            stream_conversation.replace_message(handle1, "FINAL1"),
            stream_conversation.replace_message(handle2, "FINAL2")
        )
        
        # Re-fetch messages
        msg1 = next(m for m in stream_conversation.messages if m.id == handle1.message_id)
        msg2 = next(m for m in stream_conversation.messages if m.id == handle2.message_id)
        
        assert msg1.content == "FINAL1"
        assert msg2.content == "FINAL2"
    


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
async def test_stream_conversation_integration():
    """Integration test with real Stream client (requires credentials)."""

    load_dotenv()

    # Create real client
    client = AsyncStream()
    
    # Create a test channel and user
    user = await client.create_user(id="test-user")
    channel = (await client.chat.get_or_create_channel("messaging", str(uuid.uuid4()), data=ChannelInput(created_by_id=user.id))).data.channel

    # Create conversation
    conversation = StreamConversation(
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
    await conversation.add_message(message)

    # Verify message was sent
    assert len(conversation.messages) == 1
    assert message.id in conversation.internal_ids_to_stream_ids

    # update message with replace
    await conversation.update_message(message_id=message.id, input_text="Replaced content", user_id=user.id, replace_content=True, completed=True)

    channel_data = (await client.chat.get_or_create_channel("messaging", channel.id, state=True)).data
    assert len(channel_data.messages) == 1
    assert channel_data.messages[0].text == "Replaced content"
    # Note: generating flag might not be in custom field depending on Stream API version

    # update message with delta
    await conversation.update_message(message_id=message.id, input_text=" more stuff", user_id=user.id,
                                replace_content=False, completed=True)

    channel_data = (await client.chat.get_or_create_channel("messaging", channel.id, state=True)).data
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
    await conversation.add_message(message2, completed=False)
    
    channel_data = (await client.chat.get_or_create_channel("messaging", channel.id, state=True)).data
    assert len(channel_data.messages) == 2
    assert channel_data.messages[1].text == "Still generating..."
    # Note: generating flag might not be in custom field depending on Stream API version
    
    # Test streaming handle API
    handle = await conversation.start_streaming_message(role="assistant", initial_content="Thinking")
    
    await conversation.append_to_message(handle, "...")
    
    await conversation.replace_message(handle, "The answer is 42")
    
    await conversation.complete_message(handle)
    
    channel_data = (await client.chat.get_or_create_channel("messaging", channel.id, state=True)).data
    assert len(channel_data.messages) == 3
    assert channel_data.messages[2].text == "The answer is 42"
