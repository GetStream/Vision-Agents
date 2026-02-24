"""TestResponse — data container and assertions for a single conversation turn."""

import time
from dataclasses import dataclass
from typing import Any, NoReturn

from vision_agents.testing import (
    ChatMessageEvent,
    FunctionCallEvent,
    FunctionCallOutputEvent,
    RunEvent,
)

_NOT_GIVEN = object()

_PREVIEW_MAX_LEN = 80  # max chars for content/output previews
_SELECTED_MARKER = ">>>"  # prefix for the event that caused the failure
_DEFAULT_PADDING = "   "  # prefix for all other events


@dataclass
class TestResponse:
    """Result of a single conversation turn.

    Holds the raw event list, convenient accessors (output text, function
    calls, timing), and assertion methods that use linear search.
    """

    __test__ = False

    input: str
    output: str | None
    events: list[RunEvent]
    function_calls: list[FunctionCallEvent]
    chat_messages: list[ChatMessageEvent]
    duration_ms: float

    @classmethod
    def build(
        cls,
        *,
        events: list[RunEvent],
        user_input: str,
        start_time: float,
    ) -> "TestResponse":
        """Construct a TestResponse from raw events and timing."""
        output: str | None = None
        function_calls: list[FunctionCallEvent] = []
        chat_messages: list[ChatMessageEvent] = []

        for event in events:
            if isinstance(event, ChatMessageEvent):
                chat_messages.append(event)
                if event.role == "assistant":
                    output = event.content
            elif isinstance(event, FunctionCallEvent):
                function_calls.append(event)

        return cls(
            input=user_input,
            output=output,
            events=events,
            function_calls=function_calls,
            chat_messages=chat_messages,
            duration_ms=(time.monotonic() - start_time) * 1000,
        )

    def assert_function_called(
        self,
        name: str | None = None,
        *,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        """Assert the events contain a matching ``FunctionCallEvent``.

        Scans all ``FunctionCallEvent`` entries looking for one that
        matches the given name and arguments (partial match).

        Args:
            name: Expected function name. ``None`` to skip the check.
            arguments: Expected arguments (partial match — only specified
                keys are checked).
        """
        __tracebackhide__ = True
        for event in self.function_calls:
            if name is not None and event.name != name:
                continue

            if arguments is not None and not self._arguments_match(
                event.arguments, arguments
            ):
                continue

            return

        self._raise_with_debug_info(
            "Expected FunctionCallEvent, but no matching event found."
        )

    @staticmethod
    def _arguments_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
        """Check whether *actual* contains all key/value pairs from *expected*."""
        for key, value in expected.items():
            if key not in actual or actual[key] != value:
                return False
        return True

    def assert_function_output(
        self,
        name: str,
        *,
        output: Any = _NOT_GIVEN,
        is_error: bool | None = None,
    ) -> None:
        """Assert the events contain a matching ``FunctionCallOutputEvent``.

        Scans all ``FunctionCallOutputEvent`` entries looking for one
        that matches the given name, output, and error flag.

        Args:
            name: Expected function name.
            output: Expected output value (exact match). Omit to skip.
            is_error: Expected error flag. ``None`` to skip the check.
        """
        __tracebackhide__ = True
        for event in self.events:
            if not isinstance(event, FunctionCallOutputEvent):
                continue

            if event.name != name:
                continue

            if output is not _NOT_GIVEN and event.output != output:
                continue

            if is_error is not None and event.is_error != is_error:
                continue

            return

        self._raise_with_debug_info(
            "Expected FunctionCallOutputEvent, but no matching event found."
        )

    def _raise_with_debug_info(
        self,
        message: str,
        event_index: int | None = None,
    ) -> NoReturn:
        __tracebackhide__ = True
        events_str = "\n".join(
            self._format_events(self.events, selected_index=event_index)
        )
        raise AssertionError(f"{message}\nContext:\n{events_str}")

    @classmethod
    def _truncate(cls, text: str, max_len: int = _PREVIEW_MAX_LEN) -> str:
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    @classmethod
    def _format_event(cls, event: RunEvent) -> str:
        if isinstance(event, ChatMessageEvent):
            preview = event.content[:_PREVIEW_MAX_LEN].replace("\n", "\\n")
            return f"ChatMessageEvent(role='{event.role}', content='{preview}')"
        if isinstance(event, FunctionCallEvent):
            return (
                f"FunctionCallEvent(name='{event.name}', arguments={event.arguments})"
            )
        if isinstance(event, FunctionCallOutputEvent):
            output_repr = cls._truncate(repr(event.output))
            return f"FunctionCallOutputEvent(name='{event.name}', output={output_repr}, is_error={event.is_error})"
        return repr(event)

    @classmethod
    def _format_events(
        cls,
        events: list[RunEvent],
        *,
        selected_index: int | None = None,
    ) -> list[str]:
        """Format events for debug output."""
        lines: list[str] = []
        for i, event in enumerate(events):
            marker = _SELECTED_MARKER if i == selected_index else _DEFAULT_PADDING
            lines.append(f"{marker} [{i}] {cls._format_event(event)}")
        return lines
