"""TestResponse — data container and assertions for a single conversation turn."""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from vision_agents.testing import (
    ChatMessageEvent,
    FunctionCallEvent,
    FunctionCallOutputEvent,
    RunEvent,
)

_NOT_GIVEN = object()

_PREVIEW_MAX_LEN = 80  # max chars for content/output previews


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
            if self._call_matches(event, name, arguments):
                return

        expected = self._describe_call(name, arguments)
        msg = f"Expected a {expected}, but no matching call was found."
        if self.function_calls:
            msg += f"\nFunction calls:\n{self._format_calls()}"
        raise AssertionError(msg)

    def assert_function_not_called(
        self,
        name: str | None = None,
        *,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        """Assert the events contain no matching ``FunctionCallEvent``.

        Args:
            name: Function name that must not have been called. ``None``
                to assert that no function was called at all.
            arguments: Only count calls with these arguments (partial
                match — only specified keys are checked).
        """
        __tracebackhide__ = True
        matches = [
            event
            for event in self.function_calls
            if self._call_matches(event, name, arguments)
        ]
        if not matches:
            return

        expected = self._describe_call(name, arguments)
        raise AssertionError(
            f"Expected no {expected}, but found {len(matches)}."
            f"\nFunction calls:\n{self._format_calls()}"
        )

    def assert_function_call_order(self, names: Sequence[str]) -> None:
        """Assert the named functions were called in the given relative order.

        Other calls may appear in between. Each name is matched to the next
        call with that name after the previous match, so repeating a name
        requires that many calls.

        Args:
            names: Function names in the expected order.

        Raises:
            ValueError: If *names* is empty or a single string.
        """
        __tracebackhide__ = True
        if isinstance(names, str) or not names:
            raise ValueError("names must be a sequence of at least one function name")

        actual = [event.name for event in self.function_calls]
        position = 0
        previous: str | None = None
        for expected in names:
            if expected not in actual[position:]:
                after = f" after '{previous}'" if previous is not None else ""
                msg = (
                    f"Expected function calls in order {list(names)!r}, "
                    f"but '{expected}' was not called{after}."
                    f"\nActual order: {actual!r}"
                )
                if self.function_calls:
                    msg += f"\nFunction calls:\n{self._format_calls()}"
                raise AssertionError(msg)
            position = actual.index(expected, position) + 1
            previous = expected

    @classmethod
    def _call_matches(
        cls,
        event: FunctionCallEvent,
        name: str | None,
        arguments: dict[str, Any] | None,
    ) -> bool:
        if name is not None and event.name != name:
            return False
        if arguments is not None and not cls._arguments_match(
            event.arguments, arguments
        ):
            return False
        return True

    @staticmethod
    def _arguments_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
        """Check whether *actual* contains all key/value pairs from *expected*."""
        for key, value in expected.items():
            if key not in actual or actual[key] != value:
                return False
        return True

    @staticmethod
    def _describe_call(name: str | None, arguments: dict[str, Any] | None) -> str:
        if name and arguments:
            args = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
            return f"call to '{name}({args})'"
        if name:
            return f"call to '{name}'"
        return "function call"

    def _format_calls(self) -> str:
        return "\n".join(f"   {self._format_event(fc)}" for fc in self.function_calls)

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

        error_prefix = "an error output" if is_error else "an output"
        if output is not _NOT_GIVEN:
            expected = f"{error_prefix} {output!r} from '{name}'"
        else:
            expected = f"{error_prefix} from '{name}'"

        outputs = [e for e in self.events if isinstance(e, FunctionCallOutputEvent)]
        msg = f"Expected {expected}, but no matching output was found."
        if outputs:
            lines = "\n".join(f"   {self._format_event(o)}" for o in outputs)
            msg += f"\nFunction outputs:\n{lines}"
        raise AssertionError(msg)

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
