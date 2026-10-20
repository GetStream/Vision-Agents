import asyncio
import datetime
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from vision_agents.core.edge.types import Participant

logger = logging.getLogger(__name__)


@dataclass
class Message:
    content: str
    original: Optional[Any] = None  # the original openai, claude or gemini message
    timestamp: Optional[datetime.datetime] = None
    role: Optional[str] = None
    user_id: Optional[str] = None
    id: Optional[str] = None
    # Who said it, ahead of the words the model reads. Set once a second person has spoken.
    speaker: Optional[str] = None

    def __post_init__(self):
        self.id = self.id or str(uuid.uuid4())
        self.timestamp = datetime.datetime.now()

    @property
    def named_content(self) -> str:
        """The content as the model reads it: who said it ahead of the words, when known."""
        return named(self.speaker, self.content)


def named(speaker: Optional[str], text: str) -> str:
    """``text`` with ``speaker`` ahead of it, as the router's agent writes a turn."""
    return f"{speaker}: {text}" if speaker else text


def _person(participant: Participant) -> str:
    """Who a participant is: a person who rejoins gets a new id but keeps their user."""
    return participant.user_id or participant.id


def _speaker_label(participant: Participant) -> str:
    """A participant's display name as it goes ahead of their words, empty for none.

    A line break or a colon in a display name could pass for the start of another
    person's line, so they go.
    """
    return " ".join(participant.name.replace(":", "").split())


@dataclass(frozen=True)
class _NameTag:
    """How a person whose words are in the conversation is labelled."""

    # Goes ahead of their words.
    label: str
    # The label they were given for having no display name at their first message.
    # Empty for a person who had one.
    fallback: str = ""


class ContentBuffer:
    """Manages out-of-order fragment buffering for streaming messages."""

    def __init__(self):
        self.fragments: Dict[int, str] = {}
        self.last_index = -1
        self.accumulated = ""

    def add_fragment(self, index: int, text: str):
        """Add a fragment and apply all sequential pending fragments."""
        self.fragments[index] = text
        self._apply_pending()

    def _apply_pending(self):
        """Apply all sequential fragments starting from last_index + 1."""
        while (self.last_index + 1) in self.fragments:
            self.accumulated += self.fragments.pop(self.last_index + 1)
            self.last_index += 1

    def get_accumulated(self) -> str:
        return self.accumulated

    def clear(self):
        self.fragments.clear()
        self.last_index = -1
        self.accumulated = ""


class MessageState:
    """Internal state for tracking a message's lifecycle."""

    def __init__(self, message_id: str):
        self.message_id = message_id
        self.buffer = ContentBuffer()
        self.created_in_backend = False  # Has message been sent to Stream/DB?
        self.backend_message_ids: List[
            str
        ] = []  # For chunking: multiple backend IDs per internal ID


class Conversation(ABC):
    """Base conversation class with unified message API."""

    def __init__(
        self,
        instructions: str,
        messages: List[Message],
    ):
        self.instructions = instructions
        self.messages = [m for m in messages]
        self._message_states: Dict[str, MessageState] = {}
        self._lock = asyncio.Lock()  # One lock to rule them all
        # The people, by _person, whose words are in the conversation, and their labels.
        self._speakers: dict[str, _NameTag] = {}

    def speaker_label(self, participant: Participant) -> Optional[str]:
        """The name to put ahead of ``participant``'s words, or None.

        None while nobody else has spoken, so a conversation with one person reads
        exactly as it did before names were added.
        """
        if not self._speakers.keys() - {_person(participant)}:
            return None
        return self._name_tag(participant).label

    @property
    def several_speakers(self) -> bool:
        """True once a second person has spoken, so their words carry their names."""
        return len(self._speakers) > 1

    async def send_message(
        self,
        role: str,
        user_id: str,
        content: str,
        message_id: Optional[str] = None,
        original: Any = None,
    ) -> Message:
        """Send a simple, complete message (non-streaming).

        This is a convenience method for the common case of sending a complete message.
        For streaming messages, use upsert_message() directly.

        Args:
            role: Message role (user/assistant/system)
            user_id: User ID
            content: Complete text content
            message_id: Optional ID. If None, auto-generates.
            original: Original event/object for metadata

        Returns:
            The Message object

        Examples:
            # User message
            await conv.send_message("user", "user123", "What's the weather?")

            # Assistant response
            await conv.send_message("assistant", "agent", "It's sunny!")

            # System message
            await conv.send_message("system", "system", "User joined")
        """
        return await self.upsert_message(
            role=role,
            user_id=user_id,
            content=content,
            message_id=message_id,
            completed=True,
            original=original,
        )

    async def upsert_message(
        self,
        role: str,
        user_id: str,
        content: str = "",
        message_id: Optional[str] = None,
        content_index: Optional[int] = None,
        completed: bool = True,
        replace: bool = False,
        original: Any = None,
        participant: Optional[Participant] = None,
    ) -> Message:
        """Add or update a message. Handles streaming, non-streaming, everything.

        Args:
            role: Message role (user/assistant/system)
            user_id: User ID
            content: Text content (can be partial or complete)
            message_id: Optional ID. If None, auto-generates. If provided, updates existing.
            content_index: For streaming deltas. If provided, buffers out-of-order fragments.
            completed: If True, finalizes the message. If False, keeps it as "generating".
            replace: If True, replaces all content. If False, appends/merges with deltas.
            original: Original event/object for metadata
            participant: Who said a user message. A new message from a participant
                is counted as theirs and named once somebody else has spoken.

        Returns:
            The Message object (newly created or updated)

        Examples:
            # Streaming delta
            await conv.upsert_message("assistant", "agent", "Hello", msg_id, content_index=0, completed=False)

            # Completion (replaces partial content)
            await conv.upsert_message("assistant", "agent", "Hello world!", msg_id, completed=True, replace=True)

            # Simple non-streaming
            await conv.upsert_message("user", "user123", "Hi there!")
        """
        async with self._lock:
            # Generate ID if not provided
            if message_id is None:
                message_id = str(uuid.uuid4())

            # Find or create message
            message = self._find_message(message_id)
            if message is None:
                # New message
                message = Message(
                    id=message_id,
                    role=role,
                    user_id=user_id,
                    content="",
                    original=original,
                )
                if participant is not None:
                    message.speaker = self.speaker_label(participant)
                    self._speakers[_person(participant)] = self._name_tag(participant)
                self.messages.append(message)
                state = MessageState(message_id)
                self._message_states[message_id] = state
            else:
                # Existing message - get its state
                state_or_none = self._message_states.get(message_id)
                if state_or_none is None:
                    # Message exists but no state - was already completed
                    # Ignore late updates (deltas arriving after completion)
                    logger.debug(
                        f"Message {message_id} already completed, ignoring update. "
                        f"This happens when deltas arrive after completion."
                    )
                    return message
                state = state_or_none

            # Update content
            if content_index is not None:
                # Streaming: buffer fragments in order
                state.buffer.add_fragment(content_index, content)
                message.content = state.buffer.get_accumulated()
            elif replace:
                # Replace all content
                state.buffer.clear()
                message.content = content
            else:
                # Append to existing
                message.content += content

            # Sync to backend (implementation-specific)
            await self._sync_to_backend(message, state, completed)

            # Cleanup state if completed
            if completed:
                self._message_states.pop(message_id, None)

            return message

    def _name_tag(self, participant: Participant) -> _NameTag:
        """The label ``participant``'s words go under, as the router's agent labels them.

        It is fixed by their first message, unless that message had no display name:
        such people go under "Speaker 1", "Speaker 2" and on, in the order they first
        spoke, until their name is known and replaces it. A person whose name another
        already goes under is numbered, "Alex (2)", in the order they first spoke, so the
        model tells them apart without being told their user or participant id. Mirrors
        ``speakerLocked`` in acceleration/internal/agent/agent.go (AI-1458).
        """
        person = _person(participant)
        name = _speaker_label(participant)
        current = self._speakers.get(person)
        if current is not None and (not name or current.label != current.fallback):
            return current
        taken = {tag.label for other, tag in self._speakers.items() if other != person}
        nameless = not name
        if nameless:
            # "Speaker N" is the wording Kanat chose on AI-1458 (2026-10-19).
            fallbacks = sum(1 for tag in self._speakers.values() if tag.fallback)
            name = f"Speaker {fallbacks + 1}"
        label = name
        n = 2
        while label in taken:
            label = f"{name} ({n})"
            n += 1
        if nameless:
            return _NameTag(label, fallback=label)
        return _NameTag(label, fallback=current.fallback if current else "")

    @abstractmethod
    async def _sync_to_backend(
        self, message: Message, state: MessageState, completed: bool
    ):
        """Sync message to backend storage. Implementation-specific.

        Args:
            message: The message to sync
            state: The message's internal state
            completed: If True, finalize the message. If False, mark as still generating.
        """
        pass

    async def wait_for_pending_syncs(self) -> None:
        """Wait for any backend-sync work dispatched in the background.

        Default no-op. Implementations that fire backend syncs as tasks
        (e.g. ``StreamConversation``) override this to drain those tasks
        before tests assert or before the agent shuts down.
        """

    def _find_message(self, message_id: str) -> Optional[Message]:
        """Find a message by ID."""
        return next((m for m in self.messages if m.id == message_id), None)


class InMemoryConversation(Conversation):
    """In-memory conversation (no external storage)."""

    async def _sync_to_backend(
        self, message: Message, state: MessageState, completed: bool
    ):
        """No-op for in-memory storage - message is already in self.messages."""
        pass
