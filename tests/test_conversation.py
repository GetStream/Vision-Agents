import datetime
import pytest

from vision_agents.core.agents.conversation import (
    Conversation,
    Message,
    InMemoryConversation,
)
from vision_agents.core.edge.types import Participant

ALICE = Participant(original=None, user_id="alice", id="alice-1", name="Alice")
BOB = Participant(original=None, user_id="bob", id="bob-1", name="Bob")


class TestMessage:
    """Test suite for the Message dataclass."""

    def test_message_initialization(self):
        """Test that Message initializes correctly with default timestamp."""
        message = Message(
            original={"role": "user", "content": "Hello"},
            content="Hello",
            role="user",
            user_id="test-user",
        )

        assert message.content == "Hello"
        assert message.role == "user"
        assert message.user_id == "test-user"
        assert message.timestamp is not None
        assert isinstance(message.timestamp, datetime.datetime)
        assert message.id is not None  # Auto-generated

    def test_message_custom_id(self):
        """Test that Message accepts custom ID."""
        message = Message(
            content="Hello", role="user", user_id="test-user", id="custom-id-123"
        )

        assert message.id == "custom-id-123"


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


class TestInMemoryConversation:
    """Test suite for InMemoryConversation with new unified API."""

    @pytest.fixture
    def conversation(self):
        """Create a basic InMemoryConversation instance."""
        instructions = "You are a helpful assistant."
        messages = [
            Message(original=None, content="Hello", role="user", user_id="user1"),
            Message(
                original=None,
                content="Hi there!",
                role="assistant",
                user_id="assistant",
            ),
        ]
        return InMemoryConversation(instructions, messages)

    def test_initialization(self, conversation):
        """Test InMemoryConversation initialization."""
        assert conversation.instructions == "You are a helpful assistant."
        assert len(conversation.messages) == 2

    @pytest.mark.asyncio
    async def test_send_message(self, conversation):
        """Test send_message creates a new message."""
        message = await conversation.send_message(
            role="user", user_id="user2", content="How are you?"
        )

        assert len(conversation.messages) == 3
        assert conversation.messages[-1].content == "How are you?"
        assert conversation.messages[-1].role == "user"
        assert conversation.messages[-1].user_id == "user2"
        assert message.id is not None

    @pytest.mark.asyncio
    async def test_send_message_with_custom_id(self, conversation):
        """Test send_message with custom message ID."""
        message = await conversation.send_message(
            role="assistant",
            user_id="agent",
            content="Response",
            message_id="custom-msg-id",
        )

        assert message.id == "custom-msg-id"
        assert len(conversation.messages) == 3

    @pytest.mark.asyncio
    async def test_send_message_with_original(self, conversation):
        """Test send_message stores original event."""
        original_event = {"type": "stt", "confidence": 0.95}

        message = await conversation.send_message(
            role="user",
            user_id="user1",
            content="Transcribed text",
            original=original_event,
        )

        assert message.original == original_event

    @pytest.mark.asyncio
    async def test_upsert_message_simple(self, conversation):
        """Test upsert_message for simple non-streaming case."""
        message = await conversation.upsert_message(
            role="user", user_id="user3", content="Question", completed=True
        )

        assert len(conversation.messages) == 3
        assert message.content == "Question"

    @pytest.mark.asyncio
    async def test_upsert_message_streaming_deltas(self, conversation):
        """Test upsert_message with streaming deltas."""
        msg_id = "stream-msg"

        # Delta 1
        await conversation.upsert_message(
            role="assistant",
            user_id="agent",
            content="Hello",
            message_id=msg_id,
            content_index=0,
            completed=False,
        )

        assert len(conversation.messages) == 3
        assert conversation.messages[-1].content == "Hello"

        # Delta 2
        await conversation.upsert_message(
            role="assistant",
            user_id="agent",
            content=" world",
            message_id=msg_id,
            content_index=1,
            completed=False,
        )

        assert len(conversation.messages) == 3  # Same message
        assert conversation.messages[-1].content == "Hello world"

        # Complete
        await conversation.upsert_message(
            role="assistant",
            user_id="agent",
            content="Hello world!",
            message_id=msg_id,
            completed=True,
            replace=True,
        )

        assert len(conversation.messages) == 3  # Still same message
        assert conversation.messages[-1].content == "Hello world!"

    @pytest.mark.asyncio
    async def test_upsert_message_out_of_order_deltas(self, conversation):
        """Test that out-of-order deltas are buffered correctly."""
        msg_id = "ooo-msg"

        # Send delta index 1 first
        await conversation.upsert_message(
            role="assistant",
            user_id="agent",
            content=" world",
            message_id=msg_id,
            content_index=1,
            completed=False,
        )

        # Content should be empty (waiting for index 0)
        assert conversation.messages[-1].content == ""

        # Send delta index 0
        await conversation.upsert_message(
            role="assistant",
            user_id="agent",
            content="Hello",
            message_id=msg_id,
            content_index=0,
            completed=False,
        )

        # Now both should be applied
        assert conversation.messages[-1].content == "Hello world"

    @pytest.mark.asyncio
    async def test_upsert_message_replace_vs_append(self, conversation):
        """Test replace vs append behavior."""
        msg_id = "replace-test"

        # Create initial message
        await conversation.upsert_message(
            role="assistant",
            user_id="agent",
            content="Initial",
            message_id=msg_id,
            completed=False,
        )

        # Append
        await conversation.upsert_message(
            role="assistant",
            user_id="agent",
            content=" appended",
            message_id=msg_id,
            completed=False,
            replace=False,
        )

        assert conversation.messages[-1].content == "Initial appended"

        # Replace
        await conversation.upsert_message(
            role="assistant",
            user_id="agent",
            content="Replaced",
            message_id=msg_id,
            completed=True,
            replace=True,
        )

        assert conversation.messages[-1].content == "Replaced"

    @pytest.mark.asyncio
    async def test_late_deltas_ignored_after_completion(self, conversation):
        """Test that deltas arriving after completion are ignored."""
        msg_id = "late-delta-msg"

        # Complete message first
        await conversation.upsert_message(
            role="assistant",
            user_id="agent",
            content="Final text",
            message_id=msg_id,
            completed=True,
        )

        initial_content = conversation.messages[-1].content

        # Late delta arrives (should be ignored)
        await conversation.upsert_message(
            role="assistant",
            user_id="agent",
            content="Late",
            message_id=msg_id,
            content_index=0,
            completed=False,
        )

        # Content should be unchanged
        assert conversation.messages[-1].content == initial_content

    async def test_one_person_is_never_named(self, conversation):
        # Rejoining gives Alice a new participant id; she is still one person.
        rejoined = Participant(original=None, user_id="alice", id="alice-2")
        for participant, text in ((ALICE, "Hi"), (rejoined, "Hello?")):
            await conversation.upsert_message(
                role="user", user_id="alice", content=text, participant=participant
            )

        assert [m.named_content for m in conversation.messages[2:]] == ["Hi", "Hello?"]
        assert conversation.speaker_label(ALICE) is None

    async def test_names_go_ahead_of_words_once_a_second_person_speaks(
        self, conversation
    ):
        rejoined = Participant(
            original=None, user_id="alice", id="alice-2", name="Alice"
        )
        lines = [(ALICE, "Hi"), (BOB, "Me too"), (rejoined, "Who is that?")]
        for participant, text in lines:
            await conversation.upsert_message(
                role="user",
                user_id=participant.user_id,
                content=text,
                participant=participant,
            )

        assert [m.named_content for m in conversation.messages[2:]] == [
            "Hi",
            "Bob: Me too",
            "Alice: Who is that?",
        ]
        assert [m.content for m in conversation.messages[2:]] == [
            "Hi",
            "Me too",
            "Who is that?",
        ], "what is stored and shown is still only the words"

    async def test_a_label_cannot_pass_for_another_line(self, conversation):
        await conversation.upsert_message(
            role="user", user_id="alice", content="Hi", participant=ALICE
        )
        tricky = Participant(
            original=None, user_id="bob", id="bob-1", name="Bob:\nAlice:  Smith"
        )
        blank = Participant(original=None, user_id="carol", id="carol-1", name=" :\n")

        assert conversation.speaker_label(tricky) == "Bob Alice Smith"
        assert conversation.speaker_label(blank) == "Speaker 1", (
            "a name with nothing left is no name"
        )

    @pytest.mark.parametrize(
        "lines, told",
        [
            pytest.param(
                # (user id, participant id, display name, words)
                [
                    ("u-1", "u-1", "", "hello"),
                    # No user either: the participant is all there is to tell them by.
                    ("", "session-2", "", "hi"),
                    ("u-1", "u-1", "", "again"),
                    # The first gives a name; the next one without keeps counting.
                    ("u-1", "u-1b", "Carol", "it is Carol"),
                    ("u-1", "u-1", "", "still me"),
                    ("u-3", "u-3", "", "hey"),
                ],
                [
                    "hello",
                    "Speaker 2: hi",
                    "Speaker 1: again",
                    "Carol: it is Carol",
                    "Carol: still me",
                    "Speaker 3: hey",
                ],
                id="people without a name are numbered in the order they spoke",
            ),
            pytest.param(
                [
                    ("u-1", "s-1", "Speaker 1", "hello"),
                    ("u-2", "s-2", "", "hi"),
                    ("u-2", "s-2", "Dana", "it is Dana"),
                ],
                ["hello", "Speaker 1 (2): hi", "Dana: it is Dana"],
                id="a person named like a fallback label is not mistaken for one",
            ),
            pytest.param(
                [
                    ("u-1", "s-1", "", "hello"),
                    ("u-2", "s-2", "Bob", "hi"),
                    ("u-1", "s-1", "Speaker 1", "it is me"),
                ],
                ["hello", "Bob: hi", "Speaker 1: it is me"],
                id="a name the same as their own fallback label is not numbered",
            ),
            pytest.param(
                [
                    ("u-1", "s-1", "Alex", "hello"),
                    ("u-2", "s-2", "Alex", "hi"),
                    ("u-1", "s-1", "Alexander", "me again"),
                ],
                ["hello", "Alex (2): hi", "Alex: me again"],
                id="a name is fixed and a shared one is numbered",
            ),
        ],
    )
    async def test_who_spoke_is_labelled_as_the_router_agent_labels_them(
        self,
        conversation: InMemoryConversation,
        lines: list[tuple[str, str, str, str]],
        told: list[str],
    ):
        # The router agent's rules (acceleration/internal/agent/agent.go speakerLocked,
        # AI-1458): the same lines, labelled the same way.
        for user_id, participant_id, name, text in lines:
            await conversation.upsert_message(
                role="user",
                user_id=user_id,
                content=text,
                participant=Participant(None, user_id, participant_id, name),
            )

        assert [m.named_content for m in conversation.messages[2:]] == told
