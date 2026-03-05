"""Transcript buffer for accumulating incremental speech text."""

from typing import Literal, List

TranscriptMode = Literal["delta", "replacement", "final"]


class TranscriptBuffer:
    """Buffer for accumulating transcript text from speech events.

    Supports three modes via the ``mode`` parameter on :meth:`update`:

    ``"replacement"``
        Each update replaces the current working segment in-place.

    ``"delta"``
        Each update appends text to the current working segment.

    ``"final"``
        Finalizes the current segment. If text is provided it replaces the
        segment first. An empty-text final just marks the segment as done.

    Example – replacement flow (STT partials):
        - replacement "I" → ["I"]
        - replacement "I am" → ["I am"]
        - final "I am walking" → ["I am walking"]
        - replacement "To" → ["I am walking", "To"]
        - final "To the store" → ["I am walking", "To the store"]

    Example – delta flow (realtime LLM transcripts):
        - delta "I " → ["I "]
        - delta "am " → ["I am "]
        - delta "walking" → ["I am walking"]
        - final "" → ["I am walking"]  (just finalizes)
    """

    def __init__(self):
        self._segments: List[str] = []
        self._has_pending_partial: bool = False

    def update(self, text: str, *, mode: TranscriptMode) -> None:
        """Update the buffer with new transcript text.

        Args:
            text: The transcript text.
            mode: How to apply the text.
        """
        if mode not in ("delta", "replacement", "final"):
            raise ValueError(f"Invalid transcript mode: {mode!r}")

        if mode == "delta":
            if not text:
                return
        else:
            text = text.strip()
            if not text:
                if mode == "final":
                    self._has_pending_partial = False
                return

        if mode == "delta":
            if self._has_pending_partial and self._segments:
                self._segments[-1] += text
            else:
                self._segments.append(text)
                self._has_pending_partial = True
        elif mode == "replacement":
            if self._has_pending_partial and self._segments:
                if self._segments[-1] != text:
                    self._segments[-1] = text
            else:
                if not self._segments or self._segments[-1] != text:
                    self._segments.append(text)
                    self._has_pending_partial = True
        else:
            if self._has_pending_partial and self._segments:
                self._segments[-1] = text
            else:
                if not self._segments or self._segments[-1] != text:
                    self._segments.append(text)
            self._has_pending_partial = False

    def reset(self) -> None:
        """Clear all accumulated segments."""
        self._segments.clear()
        self._has_pending_partial = False

    @property
    def has_pending(self) -> bool:
        """Whether the buffer has an unfinalized segment."""
        return self._has_pending_partial

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
