from typing import List

from vision_agents.core.stt.events import STTTranscriptEvent, STTPartialTranscriptEvent


class TranscriptBuffer:
    """
    Buffer for accumulating transcript text from STT events.

    Stores pending text in a list of strings. When new events are received,
    updates the last item if the text has an overlapping start, otherwise
    creates a new entry.
    """

    def __init__(self):
        self._segments: List[str] = []

    def update(self, event: STTTranscriptEvent | STTPartialTranscriptEvent | str) -> None:
        """
        Update the buffer from an STT event or text string.

        Args:
            event: Either an STT event or a plain text string.

        If the new text starts with the same content as the last segment,
        updates that segment. Otherwise, appends as a new segment.
        """
        text = event if isinstance(event, str) else event.text
        text = text.strip()
        if not text:
            return

        if not self._segments:
            self._segments.append(text)
            return

        last_segment = self._segments[-1]

        # Check if new text is an extension of the last segment
        if text.startswith(last_segment):
            # New text extends the last segment
            self._segments[-1] = text
        elif last_segment.startswith(text):
            # Last segment already contains this text (duplicate/stale event)
            pass
        else:
            # No overlap - start a new segment
            self._segments.append(text)

    def reset(self) -> None:
        """Clear all accumulated segments."""
        self._segments.clear()

    @property
    def segments(self) -> List[str]:
        """Return a copy of the current segments."""
        return self._segments.copy()

    @property
    def text(self) -> str:
        """Return all segments joined with spaces."""
        return " ".join(self._segments)

    def __len__(self) -> int:
        return len(self._segments)

    def __bool__(self) -> bool:
        return bool(self._segments)

