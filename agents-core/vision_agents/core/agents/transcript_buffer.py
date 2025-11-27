from typing import List

from vision_agents.core.stt.events import STTTranscriptEvent, STTPartialTranscriptEvent


class TranscriptBuffer:
    """
    Buffer for accumulating transcript text from STT events.

    Partial events update the current working segment. Final events
    finalize the segment and any subsequent events start a new one.

    Example flow:
    - Partial "I" → ["I"]
    - Partial "I am" → ["I am"]
    - Final "I am walking" → ["I am walking"]
    - Partial "To" → ["I am walking", "To"]
    - Final "To the store" → ["I am walking", "To the store"]

    It's also possible you receive corrections
    - Partial "It's damp"
    - Partial "It's a swamp"
    - Final "It's a swamp"
    """

    def __init__(self):
        self._segments: List[str] = []
        self._has_pending_partial: bool = False

    def update(
        self, event: STTTranscriptEvent | STTPartialTranscriptEvent | str
    ) -> None:
        """
        Update the buffer from an STT event or text string.

        Args:
            event: Either an STT event or a plain text string.

        Partial events update the current working segment.
        Final events (STTTranscriptEvent or strings) finalize the segment.
        """
        text = event if isinstance(event, str) else event.text
        text = text.strip()
        if not text:
            return

        is_partial = isinstance(event, STTPartialTranscriptEvent)

        if is_partial:
            if self._has_pending_partial and self._segments:
                # Update the existing partial segment (unless text matches)
                if self._segments[-1] != text:
                    self._segments[-1] = text
            else:
                # Start a new partial segment only if different from last finalized
                if not self._segments or self._segments[-1] != text:
                    self._segments.append(text)
                    self._has_pending_partial = True
        else:
            # Final event (STTTranscriptEvent or string)
            if self._has_pending_partial and self._segments:
                # Replace the partial with the final text
                self._segments[-1] = text
            else:
                # No pending partial - add as new segment only if different from last
                if not self._segments or self._segments[-1] != text:
                    self._segments.append(text)
            self._has_pending_partial = False

    def reset(self) -> None:
        """Clear all accumulated segments."""
        self._segments.clear()
        self._has_pending_partial = False

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
