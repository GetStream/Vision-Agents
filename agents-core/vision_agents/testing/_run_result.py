"""TestResponse — data container and assertions for a single conversation turn."""

import os
import time
from dataclasses import dataclass, field
from typing import Any, NoReturn, TypeVar

from vision_agents.testing._events import (
    ChatMessageEvent,
    FunctionCallEvent,
    FunctionCallOutputEvent,
    RunEvent,
)
from vision_agents.testing._judge import evaluate_intent

_evals_verbose = bool(int(os.getenv("VISION_AGENTS_EVALS_VERBOSE", "0")))

_NOT_GIVEN = object()

_T = TypeVar("_T", bound=RunEvent)


@dataclass
class TestResponse:
    """Result of a single conversation turn.

    Holds the raw event list, convenient accessors (output text, function
    calls, timing), and cursor-based assertion methods.
    """

    __test__ = False

    input: str
    output: str | None
    events: list[RunEvent]
    function_calls: list[FunctionCallEvent]
    duration_ms: float
    _judge_llm: Any = field(default=None, repr=False)
    _cursor: int = field(default=0, repr=False)

    @staticmethod
    def build(
        *,
        events: list[RunEvent],
        user_input: str,
        start_time: float,
        judge_llm: Any = None,
    ) -> "TestResponse":
        """Construct a TestResponse from raw events and timing."""
        output: str | None = None
        function_calls: list[FunctionCallEvent] = []

        for event in events:
            if isinstance(event, ChatMessageEvent) and event.role == "assistant":
                output = event.content
            elif isinstance(event, FunctionCallEvent):
                function_calls.append(event)

        return TestResponse(
            input=user_input,
            output=output,
            events=events,
            function_calls=function_calls,
            duration_ms=(time.monotonic() - start_time) * 1000,
            _judge_llm=judge_llm,
        )

    def function_called(
        self,
        name: str | None = None,
        *,
        arguments: dict[str, Any] | None = None,
    ) -> FunctionCallEvent:
        """Assert the next event is a ``FunctionCallEvent``.

        Advances the cursor to the next ``FunctionCallEvent``, checks
        name and arguments (partial match), and auto-skips the following
        ``FunctionCallOutputEvent`` if present.

        Args:
            name: Expected function name. ``None`` to skip the check.
            arguments: Expected arguments (partial match — only specified
                keys are checked).

        Returns:
            The matched ``FunctionCallEvent``.
        """
        __tracebackhide__ = True
        event = self._advance_to_type(FunctionCallEvent, "FunctionCallEvent")

        if name is not None and event.name != name:
            self._raise_with_debug_info(
                f"Expected call name '{name}', got '{event.name}'"
            )

        if arguments is not None:
            for key, value in arguments.items():
                actual = event.arguments.get(key)
                if actual != value:
                    self._raise_with_debug_info(
                        f"For argument '{key}', expected {value!r}, got {actual!r}"
                    )

        if self._cursor < len(self.events) and isinstance(
            self.events[self._cursor], FunctionCallOutputEvent
        ):
            self._cursor += 1

        return event

    def function_output(
        self,
        *,
        output: Any = _NOT_GIVEN,
        is_error: bool | None = None,
    ) -> FunctionCallOutputEvent:
        """Assert the next event is a ``FunctionCallOutputEvent``.

        Use this when you need to inspect the tool output explicitly.
        Advances the cursor to the next ``FunctionCallOutputEvent``.

        Args:
            output: Expected output value (exact match). Omit to skip.
            is_error: Expected error flag. ``None`` to skip the check.

        Returns:
            The matched ``FunctionCallOutputEvent``.
        """
        __tracebackhide__ = True
        event = self._advance_to_type(
            FunctionCallOutputEvent, "FunctionCallOutputEvent"
        )

        if output is not _NOT_GIVEN and event.output != output:
            self._raise_with_debug_info(
                f"Expected output {output!r}, got {event.output!r}"
            )

        if is_error is not None and event.is_error != is_error:
            self._raise_with_debug_info(
                f"Expected is_error={is_error}, got {event.is_error}"
            )

        return event

    async def judge(
        self,
        *,
        intent: str | None = None,
    ) -> ChatMessageEvent:
        """Assert the next event is a ``ChatMessageEvent`` and optionally judge it.

        Advances the cursor to the next ``ChatMessageEvent``. If *intent*
        is given and a judge LLM was provided, evaluates whether the
        message fulfils the intent.

        Args:
            intent: Description of what the message should accomplish.
                Requires a judge LLM to have been set on TestEval.

        Returns:
            The matched ``ChatMessageEvent``.
        """
        __tracebackhide__ = True
        event = self._advance_to_type(ChatMessageEvent, "ChatMessageEvent")

        if intent is not None:
            if self._judge_llm is None:
                raise ValueError(
                    "Cannot evaluate intent without a judge LLM. "
                    "Pass judge=<llm> to TestEval()."
                )

            success, reason = await evaluate_intent(
                llm=self._judge_llm,
                message_content=event.content,
                intent=intent,
            )

            if not success:
                self._raise_with_debug_info(f"Judgment failed: {reason}")
            elif _evals_verbose:
                preview = event.content[:30].replace("\n", "\\n")
                print(f"  judgment passed for `{preview}...`: `{reason}`")

        return event

    def no_more_events(self) -> None:
        """Assert that no further events remain after the cursor."""
        __tracebackhide__ = True
        if self._cursor < len(self.events):
            event = self.events[self._cursor]
            self._raise_with_debug_info(
                f"Expected no more events, but found: {type(event).__name__}"
            )

    def _advance_to_type(self, expected_type: type[_T], type_name: str) -> _T:
        """Advance cursor to the next event of *expected_type*, skipping others."""
        __tracebackhide__ = True
        while self._cursor < len(self.events):
            event = self.events[self._cursor]
            self._cursor += 1
            if isinstance(event, expected_type):
                return event

        self._raise_with_debug_info(
            f"Expected {type_name}, but no matching event found."
        )

    def _raise_with_debug_info(self, message: str) -> NoReturn:
        __tracebackhide__ = True
        marker = max(0, self._cursor - 1)
        events_str = "\n".join(_format_events(self.events, selected_index=marker))
        raise AssertionError(f"{message}\nContext:\n{events_str}")


def _format_events(
    events: list[RunEvent],
    *,
    selected_index: int | None = None,
) -> list[str]:
    """Format events for debug output."""
    lines: list[str] = []
    for i, event in enumerate(events):
        prefix = (
            ">>>" if (selected_index is not None and i == selected_index) else "   "
        )
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
        lines.append(line)
    return lines
