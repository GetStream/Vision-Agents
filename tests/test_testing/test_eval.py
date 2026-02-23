"""Unit tests for TestResponse assertion methods.

These tests verify the assertion API without requiring a real LLM.
Events are pre-populated via TestResponse.build() or direct construction.
"""

import time

import pytest

from vision_agents.testing import (
    ChatMessageEvent,
    FunctionCallEvent,
    FunctionCallOutputEvent,
    TestResponse,
)


def _make_response(events: list) -> TestResponse:
    """Create a TestResponse with pre-populated events for unit testing."""
    return TestResponse.build(
        events=events,
        user_input="test input",
        start_time=time.monotonic(),
    )


def _tool_call_events() -> list:
    """Typical tool-calling sequence: call -> output -> message."""
    return [
        FunctionCallEvent(name="get_weather", arguments={"location": "Tokyo"}),
        FunctionCallOutputEvent(
            name="get_weather", output={"temp": 70, "condition": "sunny"}
        ),
        ChatMessageEvent(
            role="assistant", content="The weather in Tokyo is sunny, 70F."
        ),
    ]


def _simple_events() -> list:
    """Single assistant message."""
    return [
        ChatMessageEvent(role="assistant", content="Hello! How can I help?"),
    ]


class TestFunctionCalled:
    def test_matches_name(self):
        response = _make_response(_tool_call_events())
        response.assert_function_called("get_weather")
        assert response.function_calls[0].name == "get_weather"

    def test_matches_arguments_partial(self):
        events = [
            FunctionCallEvent(
                name="search",
                arguments={"query": "hello", "limit": 10, "offset": 0},
            ),
            FunctionCallOutputEvent(name="search", output="results"),
        ]
        response = _make_response(events)
        response.assert_function_called("search", arguments={"query": "hello"})
        assert response.function_calls[0].arguments["limit"] == 10

    def test_name_mismatch_raises(self):
        response = _make_response(_tool_call_events())
        with pytest.raises(AssertionError, match="Expected call name 'wrong_tool'"):
            response.assert_function_called("wrong_tool")

    def test_argument_mismatch_raises(self):
        response = _make_response(_tool_call_events())
        with pytest.raises(AssertionError, match="argument 'location'"):
            response.assert_function_called(
                "get_weather", arguments={"location": "Berlin"}
            )

    def test_wrong_event_type_skips_to_match(self):
        events = [
            ChatMessageEvent(role="assistant", content="Thinking..."),
            FunctionCallEvent(name="search", arguments={}),
            FunctionCallOutputEvent(name="search", output="ok"),
        ]
        response = _make_response(events)
        response.assert_function_called("search")
        assert response.function_calls[0].name == "search"

    def test_no_function_call_raises(self):
        response = _make_response(_simple_events())
        with pytest.raises(AssertionError, match="Expected FunctionCallEvent"):
            response.assert_function_called("anything")

    def test_none_name_skips_name_check(self):
        response = _make_response(_tool_call_events())
        response.assert_function_called()
        assert response.function_calls[0].name == "get_weather"


class TestFunctionOutput:
    def test_matches_output(self):
        events = [
            FunctionCallOutputEvent(
                name="get_weather", output={"temp": 70, "condition": "sunny"}
            ),
        ]
        response = _make_response(events)
        response.assert_function_output(output={"temp": 70, "condition": "sunny"})

    def test_output_mismatch_raises(self):
        events = [
            FunctionCallOutputEvent(name="tool", output="actual"),
        ]
        response = _make_response(events)
        with pytest.raises(AssertionError, match="Expected output"):
            response.assert_function_output(output="wrong")

    def test_is_error_match(self):
        events = [
            FunctionCallOutputEvent(
                name="tool", output={"error": "boom"}, is_error=True
            ),
        ]
        response = _make_response(events)
        response.assert_function_output(is_error=True)

    def test_is_error_mismatch_raises(self):
        events = [
            FunctionCallOutputEvent(name="tool", output="ok", is_error=False),
        ]
        response = _make_response(events)
        with pytest.raises(AssertionError, match="Expected is_error=True"):
            response.assert_function_output(is_error=True)


class TestAssistantMessage:
    def test_finds_message(self):
        response = _make_response(_simple_events())
        event = response.assistant_message()
        assert event.content == "Hello! How can I help?"

    def test_skips_non_message_events(self):
        response = _make_response(_tool_call_events())
        event = response.assistant_message()
        assert event.content == "The weather in Tokyo is sunny, 70F."

    def test_no_message_raises(self):
        events = [
            FunctionCallEvent(name="tool", arguments={}),
        ]
        response = _make_response(events)
        with pytest.raises(AssertionError, match="Expected ChatMessageEvent"):
            response.assistant_message()


class TestFullSequence:
    def test_call_then_assistant_message(self):
        response = _make_response(_tool_call_events())
        response.assert_function_called("get_weather", arguments={"location": "Tokyo"})
        response.assistant_message()

    def test_multiple_tool_calls(self):
        events = [
            FunctionCallEvent(name="get_weather", arguments={"location": "Tokyo"}),
            FunctionCallOutputEvent(name="get_weather", output={"temp": 70}),
            FunctionCallEvent(name="get_news", arguments={"topic": "tech"}),
            FunctionCallOutputEvent(name="get_news", output=["headline1"]),
            ChatMessageEvent(role="assistant", content="Here's the info."),
        ]
        response = _make_response(events)
        assert len(response.function_calls) == 2
        response.assert_function_called("get_weather")
        assert response.function_calls[1].name == "get_news"
        assert response.function_calls[1].arguments == {"topic": "tech"}
        response.assistant_message()

    def test_explicit_output_check(self):
        events = [
            FunctionCallEvent(name="get_weather", arguments={"location": "Tokyo"}),
            FunctionCallOutputEvent(
                name="get_weather", output={"temp": 70, "condition": "sunny"}
            ),
            ChatMessageEvent(role="assistant", content="Sunny, 70F."),
        ]
        response = _make_response(events)
        response.assert_function_output(output={"temp": 70, "condition": "sunny"})


class TestTestResponse:
    def test_build_extracts_output(self):
        resp = _make_response(_tool_call_events())
        assert resp.input == "test input"
        assert resp.output == "The weather in Tokyo is sunny, 70F."
        assert len(resp.function_calls) == 1
        assert resp.function_calls[0].name == "get_weather"
        assert resp.duration_ms >= 0

    def test_build_no_assistant_message(self):
        events = [
            FunctionCallEvent(name="tool", arguments={}),
            FunctionCallOutputEvent(name="tool", output="ok"),
        ]
        resp = _make_response(events)
        assert resp.output is None
        assert len(resp.function_calls) == 1

    def test_build_simple_message(self):
        resp = _make_response(_simple_events())
        assert resp.output == "Hello! How can I help?"
        assert resp.function_calls == []


class TestErrorMessages:
    def test_includes_context_in_error(self):
        response = _make_response(_tool_call_events())
        with pytest.raises(AssertionError, match="Context:"):
            response.assert_function_called("nonexistent_tool")

    def test_includes_event_list_in_error(self):
        response = _make_response(_tool_call_events())
        with pytest.raises(AssertionError, match="FunctionCallEvent"):
            response.assert_function_called("nonexistent_tool")


class TestCallCounting:
    def test_called_times_exact(self):
        events = [
            FunctionCallEvent(name="search", arguments={"q": "a"}),
            FunctionCallOutputEvent(name="search", output="r1"),
            FunctionCallEvent(name="search", arguments={"q": "b"}),
            FunctionCallOutputEvent(name="search", output="r2"),
            ChatMessageEvent(role="assistant", content="Done."),
        ]
        response = _make_response(events)
        response.assert_function_called_times("search", 2)
        assert response.function_calls[0].arguments == {"q": "a"}
        assert response.function_calls[1].arguments == {"q": "b"}

    def test_called_times_mismatch(self):
        events = [
            FunctionCallEvent(name="search", arguments={}),
            FunctionCallOutputEvent(name="search", output="ok"),
        ]
        response = _make_response(events)
        with pytest.raises(AssertionError, match="called 1 time"):
            response.assert_function_called_times("search", 3)

    def test_not_called_passes(self):
        response = _make_response(_simple_events())
        response.assert_function_not_called("search")

    def test_not_called_raises(self):
        response = _make_response(_tool_call_events())
        with pytest.raises(
            AssertionError, match="Expected 'get_weather' not to be called"
        ):
            response.assert_function_not_called("get_weather")
