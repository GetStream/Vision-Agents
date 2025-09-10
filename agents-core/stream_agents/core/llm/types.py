from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class StandardizedTextDeltaEvent:
    content_index: int
    """The index of the content part that the text delta was added to."""

    delta: str
    """The text delta that was added."""


    type: Literal["response.output_text.delta"]
    """The type of the event. Always `response.output_text.delta`."""


    item_id: Optional[str] = None
    """The ID of the output item that the text delta was added to."""

    output_index: Optional[int] = None
    """The index of the output item that the text delta was added to."""

    sequence_number: Optional[int] = None
    """The sequence number for this event."""