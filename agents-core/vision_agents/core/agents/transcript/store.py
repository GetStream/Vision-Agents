"""Transcript store for tracking buffered transcripts and message IDs."""

import dataclasses
import uuid
from typing import get_args

from .buffer import TranscriptBuffer, TranscriptMode

_VALID_MODES = get_args(TranscriptMode)


@dataclasses.dataclass(frozen=True, slots=True)
class TranscriptUpdate:
    """Result of updating a transcript entry.

    The ``mode`` field indicates how ``text`` should be interpreted:

    ``"delta"`` – ``text`` is a new fragment to append.
    ``"replacement"`` – ``text`` is the full accumulated text so far.
    ``"final"`` – ``text`` is the full accumulated text; the entry is complete.
    """

    message_id: str
    user_id: str
    text: str
    mode: TranscriptMode


class TranscriptStore:
    """Tracks transcript buffers and message IDs for active speakers.

    Manages separate entries for each user participant (keyed by
    participant.id) and a single entry for the agent. Provides flush
    helpers that callers use to implement cross-speaker finalization.
    """

    def __init__(self, agent_user_id: str):
        if not agent_user_id:
            raise ValueError("agent_user_id must be a non-empty string")
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
        if mode not in _VALID_MODES:
            raise ValueError(f"Invalid transcript mode: {mode!r}")
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
                message_id=msg_id, user_id=uid, text=buffer.text, mode=mode
            )
        elif mode == "replacement":
            return TranscriptUpdate(
                message_id=msg_id, user_id=uid, text=buffer.text, mode=mode
            )
        else:
            return TranscriptUpdate(
                message_id=msg_id, user_id=uid, text=text, mode=mode
            )

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
        if mode not in _VALID_MODES:
            raise ValueError(f"Invalid transcript mode: {mode!r}")
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
                mode=mode,
            )
        elif mode == "replacement":
            return TranscriptUpdate(
                message_id=msg_id,
                user_id=self._agent_user_id,
                text=buffer.text,
                mode=mode,
            )
        else:
            return TranscriptUpdate(
                message_id=msg_id,
                user_id=self._agent_user_id,
                text=text,
                mode=mode,
            )

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
                        mode="final",
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
            mode="final",
        )
