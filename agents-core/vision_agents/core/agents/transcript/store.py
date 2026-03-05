"""Transcript store for tracking buffered transcripts and message IDs."""

import dataclasses
import uuid

from .buffer import TranscriptBuffer, TranscriptMode


@dataclasses.dataclass(frozen=True, slots=True)
class TranscriptUpdate:
    """Result of updating a transcript entry."""

    message_id: str
    user_id: str
    text: str
    completed: bool


class TranscriptStore:
    """Tracks transcript buffers and message IDs for active speakers.

    Manages separate entries for each user participant (keyed by
    participant.id) and a single entry for the agent. Handles cross-speaker
    finalization: starting a user transcript finalizes the pending agent
    transcript, and vice versa.
    """

    def __init__(self, agent_user_id: str):
        self._agent_user_id = agent_user_id
        self._users: dict[str, tuple[str, str, TranscriptBuffer]] = {}
        self._agent: tuple[str, TranscriptBuffer] | None = None

    def update_user_transcript(
        self,
        *,
        participant_id: str,
        user_id: str,
        text: str,
        mode: TranscriptMode,
    ) -> TranscriptUpdate | None:
        """Update a user transcript. Returns update info, or None if skipped."""
        entry = self._users.get(participant_id)
        if entry is None:
            if not text:
                return None
            entry = (str(uuid.uuid4()), user_id, TranscriptBuffer())
            self._users[participant_id] = entry

        msg_id, uid, buffer = entry
        buffer.update(text, mode=mode)

        if not buffer:
            return None

        if mode == "final":
            self._users.pop(participant_id, None)
            return TranscriptUpdate(
                message_id=msg_id, user_id=uid, text=buffer.text, completed=True
            )
        elif mode == "replacement":
            return TranscriptUpdate(
                message_id=msg_id, user_id=uid, text=buffer.text, completed=False
            )
        elif mode == "delta":
            return TranscriptUpdate(
                message_id=msg_id, user_id=uid, text=text, completed=False
            )
        else:
            raise ValueError(f"Invalid transcript update mode: {mode}")

    def get_buffer(
        self, *, participant_id: str, user_id: str
    ) -> TranscriptBuffer | None:
        """Return the transcript buffer for a participant.

        Picks the agent buffer when user_id matches the agent, otherwise
        looks up the user buffer by participant_id.
        """
        if user_id == self._agent_user_id:
            if self._agent:
                _, buffer = self._agent
                return buffer
            return None
        entry = self._users.get(participant_id)
        if entry is None:
            return None
        _, _, buffer = entry
        return buffer

    def update_agent_transcript(
        self, *, text: str, mode: TranscriptMode
    ) -> TranscriptUpdate | None:
        """Update the agent transcript. Returns update info, or None if skipped."""
        entry = self._agent
        if entry is None:
            if not text:
                return None
            entry = (str(uuid.uuid4()), TranscriptBuffer())
            self._agent = entry

        msg_id, buffer = entry
        buffer.update(text, mode=mode)

        if not buffer:
            return None

        if mode == "final":
            self._agent = None
            return TranscriptUpdate(
                message_id=msg_id,
                user_id=self._agent_user_id,
                text=buffer.text,
                completed=True,
            )
        elif mode == "replacement":
            return TranscriptUpdate(
                message_id=msg_id,
                user_id=self._agent_user_id,
                text=buffer.text,
                completed=False,
            )
        elif mode == "delta":
            return TranscriptUpdate(
                message_id=msg_id,
                user_id=self._agent_user_id,
                text=text,
                completed=False,
            )
        else:
            raise ValueError(f"Invalid transcript update mode: {mode}")

    def flush_users_transcripts(self) -> list[TranscriptUpdate]:
        """Return pending user transcripts for finalization and clear them."""
        results = []
        for msg_id, user_id, buffer in self._users.values():
            if buffer:
                results.append(
                    TranscriptUpdate(
                        message_id=msg_id,
                        user_id=user_id,
                        text=buffer.text,
                        completed=True,
                    )
                )
        self._users.clear()
        return results

    def flush_agent_transcript(self) -> TranscriptUpdate | None:
        """Return pending agent transcript for finalization and clear it."""
        if self._agent is None:
            return None
        msg_id, buffer = self._agent
        self._agent = None
        if not buffer:
            return None
        return TranscriptUpdate(
            message_id=msg_id,
            user_id=self._agent_user_id,
            text=buffer.text,
            completed=True,
        )
