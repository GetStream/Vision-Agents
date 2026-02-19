"""Unit tests for RunResult and assertion classes.

These tests verify the fluent assertion API without requiring a real LLM.
"""

import pytest

from vision_agents.testing._events import (
    ChatMessageEvent,
    FunctionCallEvent,
    FunctionCallOutputEvent,
)
from vision_agents.testing._run_result import (
    ChatMessageAssert,
    EventAssert,
    FunctionCallAssert,
    FunctionCallOutputAssert,
    RunResult,
)


def _make_result() -> RunResult:
    """Build a RunResult with a typical tool-calling sequence."""
    return RunResult(
        events=[
            FunctionCallEvent(name="get_weather", arguments={"location": "Tokyo"}),
            FunctionCallOutputEvent(
                name="get_weather", output={"temp": 70, "condition": "sunny"}
            ),
            ChatMessageEvent(role="assistant", content="The weather in Tokyo is sunny, 70F."),
        ],
        user_input="What's the weather in Tokyo?",
    )


def _make_simple_result() -> RunResult:
    return RunResult(
        events=[
            ChatMessageEvent(role="assistant", content="Hello! How can I help?"),
        ],
        user_input="Hello",
    )


# ---------- next_event / is_* ------------------------------------------


class TestNextEvent:
    def test_sequential_navigation(self):
        result = _make_result()
        result.expect.next_event().is_function_call(name="get_weather")
        result.expect.next_event().is_function_call_output()
        result.expect.next_event().is_message(role="assistant")
        result.expect.no_more_events()

    def test_is_function_call_with_arguments(self):
        result = _make_result()
        fc = result.expect.next_event().is_function_call(
            name="get_weather", arguments={"location": "Tokyo"}
        )
        assert isinstance(fc, FunctionCallAssert)
        assert fc.event().name == "get_weather"

    def test_is_message_role_mismatch(self):
        result = _make_simple_result()
        with pytest.raises(AssertionError, match="Expected role 'user'"):
            result.expect.next_event().is_message(role="user")

    def test_is_message_wrong_event_type(self):
        result = _make_result()
        with pytest.raises(AssertionError, match="Expected ChatMessageEvent"):
            result.expect.next_event().is_message()

    def test_next_event_with_type_filter(self):
        result = _make_result()
        msg = result.expect.next_event(type="message")
        assert isinstance(msg, ChatMessageAssert)
        assert msg.event().content == "The weather in Tokyo is sunny, 70F."

    def test_next_event_exhausted(self):
        result = _make_simple_result()
        result.expect.next_event()
        with pytest.raises(AssertionError, match="Expected another event"):
            result.expect.next_event()


# ---------- no_more_events ---------------------------------------------


class TestNoMoreEvents:
    def test_pass(self):
        result = _make_simple_result()
        result.expect.next_event()
        result.expect.no_more_events()

    def test_fail(self):
        result = _make_simple_result()
        with pytest.raises(AssertionError, match="Expected no more events"):
            result.expect.no_more_events()


# ---------- skip_next ---------------------------------------------------


class TestSkipNext:
    def test_skip_one(self):
        result = _make_result()
        result.expect.skip_next()
        result.expect.next_event().is_function_call_output()

    def test_skip_multiple(self):
        result = _make_result()
        result.expect.skip_next(2)
        result.expect.next_event().is_message(role="assistant")
        result.expect.no_more_events()

    def test_skip_past_end(self):
        result = _make_simple_result()
        with pytest.raises(AssertionError, match="skip"):
            result.expect.skip_next(5)


# ---------- indexed access ----------------------------------------------


class TestIndexedAccess:
    def test_positive_index(self):
        result = _make_result()
        result.expect[0].is_function_call(name="get_weather")
        result.expect[2].is_message(role="assistant")

    def test_negative_index(self):
        result = _make_result()
        result.expect[-1].is_message(role="assistant")

    def test_out_of_range(self):
        result = _make_simple_result()
        with pytest.raises(AssertionError, match="out of range"):
            result.expect[5]


# ---------- sliced access & contains_* ---------------------------------


class TestSlicedAccess:
    def test_contains_message(self):
        result = _make_result()
        msg = result.expect[:].contains_message(role="assistant")
        assert isinstance(msg, ChatMessageAssert)

    def test_contains_function_call(self):
        result = _make_result()
        fc = result.expect[:].contains_function_call(name="get_weather")
        assert isinstance(fc, FunctionCallAssert)

    def test_contains_function_call_output(self):
        result = _make_result()
        fco = result.expect[:].contains_function_call_output()
        assert isinstance(fco, FunctionCallOutputAssert)

    def test_contains_message_not_found(self):
        result = _make_result()
        with pytest.raises(AssertionError, match="No ChatMessageEvent"):
            result.expect[:].contains_message(role="user")

    def test_contains_function_call_not_found(self):
        result = _make_simple_result()
        with pytest.raises(AssertionError, match="No FunctionCallEvent"):
            result.expect[:].contains_function_call(name="nonexistent")

    def test_range_slice(self):
        result = _make_result()
        result.expect[0:2].contains_function_call(name="get_weather")

    def test_shorthand_contains(self):
        result = _make_result()
        result.expect.contains_message(role="assistant")
        result.expect.contains_function_call(name="get_weather")


# ---------- is_function_call_output assertions --------------------------


class TestFunctionCallOutputAssert:
    def test_output_match(self):
        result = _make_result()
        result.expect.skip_next()
        result.expect.next_event().is_function_call_output(
            output={"temp": 70, "condition": "sunny"}
        )

    def test_output_mismatch(self):
        result = _make_result()
        result.expect.skip_next()
        with pytest.raises(AssertionError, match="Expected output"):
            result.expect.next_event().is_function_call_output(output="wrong")

    def test_is_error_check(self):
        err_result = RunResult(
            events=[
                FunctionCallOutputEvent(
                    name="tool", output={"error": "boom"}, is_error=True
                ),
            ]
        )
        err_result.expect.next_event().is_function_call_output(is_error=True)


# ---------- argument partial matching -----------------------------------


class TestArgumentMatching:
    def test_partial_arguments(self):
        result = RunResult(
            events=[
                FunctionCallEvent(
                    name="search",
                    arguments={"query": "hello", "limit": 10, "offset": 0},
                ),
            ]
        )
        result.expect.next_event().is_function_call(
            name="search", arguments={"query": "hello"}
        )

    def test_argument_mismatch(self):
        result = RunResult(
            events=[
                FunctionCallEvent(name="search", arguments={"query": "hello"}),
            ]
        )
        with pytest.raises(AssertionError, match="argument 'query'"):
            result.expect.next_event().is_function_call(
                name="search", arguments={"query": "world"}
            )


# ---------- event() accessor -------------------------------------------


class TestEventAccessor:
    def test_message_event_accessor(self):
        result = _make_simple_result()
        msg = result.expect.next_event().is_message()
        assert msg.event().content == "Hello! How can I help?"
        assert msg.event().role == "assistant"

    def test_function_call_event_accessor(self):
        result = _make_result()
        fc = result.expect.next_event().is_function_call()
        assert fc.event().name == "get_weather"
        assert fc.event().arguments == {"location": "Tokyo"}

    def test_raw_event_accessor(self):
        result = _make_result()
        ev = result.expect.next_event()
        assert isinstance(ev, EventAssert)
        assert ev.event().type == "function_call"
