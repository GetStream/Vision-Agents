"""RunResult and assertion classes for testing Vision-Agents.

Provides a fluent assertion API for verifying agent behavior::

    result.expect.next_event().is_message(role="assistant")
    result.expect.next_event().is_function_call(name="get_weather")
    result.expect.no_more_events()
"""

from __future__ import annotations

import contextlib
import functools
import os
from typing import Any, Literal, overload

from vision_agents.testing._events import (
    ChatMessageEvent,
    FunctionCallEvent,
    FunctionCallOutputEvent,
    RunEvent,
)

_evals_verbose = bool(int(os.getenv("VISION_AGENTS_EVALS_VERBOSE", "0")))

_NOT_GIVEN = object()


def _format_events(
    events: list[RunEvent],
    *,
    selected_index: int | None = None,
) -> list[str]:
    """Format events for debug output."""
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


# ---------------------------------------------------------------------------
# RunResult
# ---------------------------------------------------------------------------


class RunResult:
    """Result of a single conversation turn.

    Holds captured events and provides an assertion interface via ``.expect``.
    """

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

    @functools.cached_property
    def expect(self) -> RunAssert:
        """Fluent assertion interface for verifying events."""
        if _evals_verbose:
            events_str = "\n    ".join(_format_events(self._events))
            print(
                f"\n+ RunResult(\n"
                f"   user_input=`{self._user_input}`\n"
                f"   events:\n    {events_str}\n"
                f")"
            )
        return RunAssert(self)


# ---------------------------------------------------------------------------
# RunAssert — cursor-based navigation over events
# ---------------------------------------------------------------------------


class RunAssert:
    """Cursor-based assertion navigator over RunResult events.

    Supports sequential navigation, indexed access, slicing, and search.
    """

    def __init__(self, run_result: RunResult) -> None:
        self._events_list = run_result.events
        self._current_index = 0

    # -- indexed / sliced access ------------------------------------------

    @overload
    def __getitem__(self, index: int) -> EventAssert: ...

    @overload
    def __getitem__(self, s: slice) -> EventRangeAssert: ...

    def __getitem__(self, key: int | slice) -> EventAssert | EventRangeAssert:
        __tracebackhide__ = True
        if isinstance(key, slice):
            return EventRangeAssert(self._events_list[key], self, key)
        if isinstance(key, int):
            idx = key if key >= 0 else key + len(self._events_list)
            if not (0 <= idx < len(self._events_list)):
                self._raise_with_debug_info(
                    f"Index {key} out of range (total events: {len(self._events_list)})",
                    index=idx,
                )
            return EventAssert(self._events_list[idx], self, idx)
        raise TypeError(f"Indices must be int or slice, not {type(key).__name__}")

    # -- sequential navigation --------------------------------------------

    @overload
    def next_event(self, *, type: None = None) -> EventAssert: ...

    @overload
    def next_event(self, *, type: Literal["message"]) -> ChatMessageAssert: ...

    @overload
    def next_event(self, *, type: Literal["function_call"]) -> FunctionCallAssert: ...

    @overload
    def next_event(
        self, *, type: Literal["function_call_output"]
    ) -> FunctionCallOutputAssert: ...

    def next_event(
        self,
        *,
        type: Literal["message", "function_call", "function_call_output"] | None = None,
    ) -> EventAssert | ChatMessageAssert | FunctionCallAssert | FunctionCallOutputAssert:
        """Advance cursor to the next event, optionally filtering by type.

        When *type* is given, non-matching events are skipped.
        Raises ``AssertionError`` if no matching event is found.
        """
        __tracebackhide__ = True
        while True:
            ev_assert = self._current_event()
            self._current_index += 1
            if type is None or ev_assert.event().type == type:
                break

        if type == "message":
            return ev_assert.is_message()
        if type == "function_call":
            return ev_assert.is_function_call()
        if type == "function_call_output":
            return ev_assert.is_function_call_output()
        return ev_assert

    def skip_next(self, count: int = 1) -> RunAssert:
        """Skip *count* events without assertion."""
        __tracebackhide__ = True
        for i in range(count):
            if self._current_index >= len(self._events_list):
                self._raise_with_debug_info(
                    f"Tried to skip {count} event(s), but only {i} were available."
                )
            self._current_index += 1
        return self

    def no_more_events(self) -> None:
        """Assert that no further events remain."""
        __tracebackhide__ = True
        if self._current_index < len(self._events_list):
            event = self._events_list[self._current_index]
            self._raise_with_debug_info(
                f"Expected no more events, but found: {type(event).__name__}"
            )

    # -- search (order-agnostic) ------------------------------------------

    def contains_message(
        self, *, role: Any = _NOT_GIVEN
    ) -> ChatMessageAssert:
        """Search all events for a matching message."""
        __tracebackhide__ = True
        return self[:].contains_message(role=role)

    def contains_function_call(
        self,
        *,
        name: Any = _NOT_GIVEN,
        arguments: Any = _NOT_GIVEN,
    ) -> FunctionCallAssert:
        """Search all events for a matching function call."""
        __tracebackhide__ = True
        return self[:].contains_function_call(name=name, arguments=arguments)

    def contains_function_call_output(
        self,
        *,
        output: Any = _NOT_GIVEN,
        is_error: Any = _NOT_GIVEN,
    ) -> FunctionCallOutputAssert:
        """Search all events for a matching function call output."""
        __tracebackhide__ = True
        return self[:].contains_function_call_output(output=output, is_error=is_error)

    # -- internals --------------------------------------------------------

    def _current_event(self) -> EventAssert:
        __tracebackhide__ = True
        if self._current_index >= len(self._events_list):
            self._raise_with_debug_info("Expected another event, but none left.")
        return self[self._current_index]

    def _raise_with_debug_info(
        self, message: str, index: int | None = None
    ) -> None:
        __tracebackhide__ = True
        marker = self._current_index if index is None else index
        events_str = "\n".join(_format_events(self._events_list, selected_index=marker))
        raise AssertionError(f"{message}\nContext:\n{events_str}")


# ---------------------------------------------------------------------------
# EventAssert — type checking for a single event
# ---------------------------------------------------------------------------


class EventAssert:
    """Assertion helper for a single event."""

    def __init__(self, event: RunEvent, parent: RunAssert, index: int) -> None:
        self._event = event
        self._parent = parent
        self._index = index

    def event(self) -> RunEvent:
        """Return the underlying event."""
        return self._event

    def is_message(self, *, role: Any = _NOT_GIVEN) -> ChatMessageAssert:
        """Assert this event is a ``ChatMessageEvent``."""
        __tracebackhide__ = True
        if not isinstance(self._event, ChatMessageEvent):
            self._raise(
                f"Expected ChatMessageEvent, got {type(self._event).__name__}"
            )
        assert isinstance(self._event, ChatMessageEvent)  # for type narrowing
        if role is not _NOT_GIVEN and self._event.role != role:
            self._raise(f"Expected role '{role}', got '{self._event.role}'")
        return ChatMessageAssert(self._event, self._parent, self._index)

    def is_function_call(
        self,
        *,
        name: Any = _NOT_GIVEN,
        arguments: Any = _NOT_GIVEN,
    ) -> FunctionCallAssert:
        """Assert this event is a ``FunctionCallEvent``."""
        __tracebackhide__ = True
        if not isinstance(self._event, FunctionCallEvent):
            self._raise(
                f"Expected FunctionCallEvent, got {type(self._event).__name__}"
            )
        assert isinstance(self._event, FunctionCallEvent)
        if name is not _NOT_GIVEN and self._event.name != name:
            self._raise(f"Expected call name '{name}', got '{self._event.name}'")
        if arguments is not _NOT_GIVEN:
            for key, value in arguments.items():
                actual = self._event.arguments.get(key)
                if actual != value:
                    self._raise(
                        f"For argument '{key}', expected {value!r}, got {actual!r}"
                    )
        return FunctionCallAssert(self._event, self._parent, self._index)

    def is_function_call_output(
        self,
        *,
        output: Any = _NOT_GIVEN,
        is_error: Any = _NOT_GIVEN,
    ) -> FunctionCallOutputAssert:
        """Assert this event is a ``FunctionCallOutputEvent``."""
        __tracebackhide__ = True
        if not isinstance(self._event, FunctionCallOutputEvent):
            self._raise(
                f"Expected FunctionCallOutputEvent, got {type(self._event).__name__}"
            )
        assert isinstance(self._event, FunctionCallOutputEvent)
        if output is not _NOT_GIVEN and self._event.output != output:
            self._raise(f"Expected output {output!r}, got {self._event.output!r}")
        if is_error is not _NOT_GIVEN and self._event.is_error != is_error:
            self._raise(f"Expected is_error={is_error}, got {self._event.is_error}")
        return FunctionCallOutputAssert(self._event, self._parent, self._index)

    def _raise(self, message: str) -> None:
        __tracebackhide__ = True
        self._parent._raise_with_debug_info(message, index=self._index)


# ---------------------------------------------------------------------------
# EventRangeAssert — search within a slice of events
# ---------------------------------------------------------------------------


class EventRangeAssert:
    """Assertion helper for a range/slice of events."""

    def __init__(
        self, events: list[RunEvent], parent: RunAssert, rng: slice
    ) -> None:
        self._events = events
        self._parent = parent
        self._rng = rng

    def contains_message(
        self, *, role: Any = _NOT_GIVEN
    ) -> ChatMessageAssert:
        __tracebackhide__ = True
        start = self._rng.start or 0
        for idx, ev in enumerate(self._events):
            candidate = EventAssert(ev, self._parent, start + idx)
            with contextlib.suppress(AssertionError):
                return candidate.is_message(role=role)
        self._parent._raise_with_debug_info(
            f"No ChatMessageEvent matching criteria found in range"
        )
        raise RuntimeError("unreachable")

    def contains_function_call(
        self,
        *,
        name: Any = _NOT_GIVEN,
        arguments: Any = _NOT_GIVEN,
    ) -> FunctionCallAssert:
        __tracebackhide__ = True
        start = self._rng.start or 0
        for idx, ev in enumerate(self._events):
            candidate = EventAssert(ev, self._parent, start + idx)
            with contextlib.suppress(AssertionError):
                return candidate.is_function_call(name=name, arguments=arguments)
        self._parent._raise_with_debug_info(
            f"No FunctionCallEvent matching criteria found in range"
        )
        raise RuntimeError("unreachable")

    def contains_function_call_output(
        self,
        *,
        output: Any = _NOT_GIVEN,
        is_error: Any = _NOT_GIVEN,
    ) -> FunctionCallOutputAssert:
        __tracebackhide__ = True
        start = self._rng.start or 0
        for idx, ev in enumerate(self._events):
            candidate = EventAssert(ev, self._parent, start + idx)
            with contextlib.suppress(AssertionError):
                return candidate.is_function_call_output(output=output, is_error=is_error)
        self._parent._raise_with_debug_info(
            f"No FunctionCallOutputEvent matching criteria found in range"
        )
        raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Type-specific assertion classes
# ---------------------------------------------------------------------------


class ChatMessageAssert:
    """Assertion for a ``ChatMessageEvent``, with ``judge()`` support."""

    def __init__(
        self, event: ChatMessageEvent, parent: RunAssert, index: int
    ) -> None:
        self._event = event
        self._parent = parent
        self._index = index

    def event(self) -> ChatMessageEvent:
        """Return the underlying event."""
        return self._event

    async def judge(
        self,
        llm: Any,
        *,
        intent: str,
    ) -> ChatMessageAssert:
        """Evaluate whether the message fulfils *intent* using an LLM.

        Args:
            llm: LLM instance for evaluation. Should be a **separate**
                instance from the agent's LLM to avoid polluting
                conversation history.
            intent: Description of what the message should accomplish.

        Returns:
            Self for chaining.

        Raises:
            AssertionError: If the message does not fulfil the intent.
        """
        __tracebackhide__ = True
        from vision_agents.testing._judge import evaluate_intent

        success, reason = await evaluate_intent(
            llm=llm,
            message_content=self._event.content,
            intent=intent,
        )

        if not success:
            self._parent._raise_with_debug_info(
                f"Judgment failed: {reason}", index=self._index
            )
        elif _evals_verbose:
            preview = self._event.content[:30].replace("\n", "\\n")
            print(f"- Judgment succeeded for `{preview}...`: `{reason}`")

        return self

    def _raise(self, message: str) -> None:
        __tracebackhide__ = True
        self._parent._raise_with_debug_info(message, index=self._index)


class FunctionCallAssert:
    """Assertion for a ``FunctionCallEvent``."""

    def __init__(
        self, event: FunctionCallEvent, parent: RunAssert, index: int
    ) -> None:
        self._event = event
        self._parent = parent
        self._index = index

    def event(self) -> FunctionCallEvent:
        """Return the underlying event."""
        return self._event


class FunctionCallOutputAssert:
    """Assertion for a ``FunctionCallOutputEvent``."""

    def __init__(
        self, event: FunctionCallOutputEvent, parent: RunAssert, index: int
    ) -> None:
        self._event = event
        self._parent = parent
        self._index = index

    def event(self) -> FunctionCallOutputEvent:
        """Return the underlying event."""
        return self._event
