"""RunResult — data container for a single conversation turn."""

from __future__ import annotations

from vision_agents.testing._events import (
    ChatMessageEvent,
    FunctionCallEvent,
    FunctionCallOutputEvent,
    RunEvent,
)


class RunResult:
    """Result of a single conversation turn."""

    def __init__(
        self,
        events: list[RunEvent],
        user_input: str | None = None,
    ) -> None:
        self._events = events
        self._user_input = user_input

    @property
    def events(self) -> list[RunEvent]:
        """All captured events in chronological order."""
        return self._events

    @property
    def user_input(self) -> str | None:
        return self._user_input


def _format_events(
    events: list[RunEvent],
    *,
    selected_index: int | None = None,
) -> list[str]:
    """Format events for debug output. Used by TestEval assertions."""
    lines: list[str] = []
    for i, event in enumerate(events):
        prefix = ">>>" if (selected_index is not None and i == selected_index) else "   "
        if isinstance(event, ChatMessageEvent):
            preview = event.content[:80].replace("\n", "\\n")
            line = f"{prefix} [{i}] ChatMessageEvent(role='{event.role}', content='{preview}')"
        elif isinstance(event, FunctionCallEvent):
            line = f"{prefix} [{i}] FunctionCallEvent(name='{event.name}', arguments={event.arguments})"
        elif isinstance(event, FunctionCallOutputEvent):
            output_repr = repr(event.output)
            if len(output_repr) > 80:
                output_repr = output_repr[:77] + "..."
            line = f"{prefix} [{i}] FunctionCallOutputEvent(name='{event.name}', output={output_repr}, is_error={event.is_error})"
        else:
            line = f"{prefix} [{i}] {event}"
        lines.append(line)
    return lines
