"""Unit tests for TestResponse assertion methods.

These tests verify the assertion API without requiring a real LLM.
Events are pre-populated via TestResponse.build() or direct construction.
"""

import time

import pytest

from vision_agents.testing._events import (
    ChatMessageEvent,
    FunctionCallEvent,
    FunctionCallOutputEvent,
)
from vision_agents.testing._run_result import TestResponse


def _make_response(events: list, judge_llm: object | None = None) -> TestResponse:
    """Create a TestResponse with pre-populated events for unit testing."""
    return TestResponse.build(
        events=events,
        user_input="test input",
        start_time=time.monotonic(),
        judge_llm=judge_llm,
    )


def _tool_call_events() -> list:
    """Typical tool-calling sequence: call -> output -> message."""
    return [
        FunctionCallEvent(name="get_weather", arguments={"location": "Tokyo"}),
        FunctionCallOutputEvent(
            name="get_weather", output={"temp": 70, "condition": "sunny"}
        ),
        ChatMessageEvent(role="assistant", content="The weather in Tokyo is sunny, 70F."),
    ]


def _simple_events() -> list:
    """Single assistant message."""
    return [
        ChatMessageEvent(role="assistant", content="Hello! How can I help?"),
    ]


class TestFunctionCalled:
    def test_matches_name(self):
        response = _make_response(_tool_call_events())
        event = response.function_called("get_weather")
        assert event.name == "get_weather"

    def test_matches_arguments_partial(self):
        events = [
            FunctionCallEvent(
                name="search",
                arguments={"query": "hello", "limit": 10, "offset": 0},
            ),
            FunctionCallOutputEvent(name="search", output="results"),
        ]
        response = _make_response(events)
        event = response.function_called("search", arguments={"query": "hello"})
        assert event.arguments["limit"] == 10

    def test_name_mismatch_raises(self):
        response = _make_response(_tool_call_events())
        with pytest.raises(AssertionError, match="Expected call name 'wrong_tool'"):
            response.function_called("wrong_tool")

    def test_argument_mismatch_raises(self):
        response = _make_response(_tool_call_events())
        with pytest.raises(AssertionError, match="argument 'location'"):
            response.function_called("get_weather", arguments={"location": "Berlin"})

    def test_wrong_event_type_skips_to_match(self):
        events = [
            ChatMessageEvent(role="assistant", content="Thinking..."),
            FunctionCallEvent(name="search", arguments={}),
            FunctionCallOutputEvent(name="search", output="ok"),
        ]
        response = _make_response(events)
        event = response.function_called("search")
        assert event.name == "search"

    def test_no_function_call_raises(self):
        response = _make_response(_simple_events())
        with pytest.raises(AssertionError, match="Expected FunctionCallEvent"):
            response.function_called("anything")

    async def test_auto_skips_function_call_output(self):
        response = _make_response(_tool_call_events())
        response.function_called("get_weather")
        event = await response.judge()
        assert isinstance(event, ChatMessageEvent)

    def test_none_name_skips_name_check(self):
        response = _make_response(_tool_call_events())
        event = response.function_called()
        assert event.name == "get_weather"


class TestFunctionOutput:
    def test_matches_output(self):
        events = [
            FunctionCallOutputEvent(
                name="get_weather", output={"temp": 70, "condition": "sunny"}
            ),
        ]
        response = _make_response(events)
        event = response.function_output(
            output={"temp": 70, "condition": "sunny"}
        )
        assert event.name == "get_weather"

    def test_output_mismatch_raises(self):
        events = [
            FunctionCallOutputEvent(name="tool", output="actual"),
        ]
        response = _make_response(events)
        with pytest.raises(AssertionError, match="Expected output"):
            response.function_output(output="wrong")

    def test_is_error_match(self):
        events = [
            FunctionCallOutputEvent(
                name="tool", output={"error": "boom"}, is_error=True
            ),
        ]
        response = _make_response(events)
        event = response.function_output(is_error=True)
        assert event.is_error is True

    def test_is_error_mismatch_raises(self):
        events = [
            FunctionCallOutputEvent(name="tool", output="ok", is_error=False),
        ]
        response = _make_response(events)
        with pytest.raises(AssertionError, match="Expected is_error=True"):
            response.function_output(is_error=True)


class TestJudge:
    async def test_finds_message(self):
        response = _make_response(_simple_events())
        event = await response.judge()
        assert event.content == "Hello! How can I help?"

    async def test_skips_non_message_events(self):
        response = _make_response(_tool_call_events())
        event = await response.judge()
        assert event.content == "The weather in Tokyo is sunny, 70F."

    async def test_no_message_raises(self):
        events = [
            FunctionCallEvent(name="tool", arguments={}),
        ]
        response = _make_response(events)
        with pytest.raises(AssertionError, match="Expected ChatMessageEvent"):
            await response.judge()

    async def test_intent_without_judge_raises(self):
        response = _make_response(_simple_events())
        with pytest.raises(ValueError, match="Cannot evaluate intent"):
            await response.judge(intent="Friendly greeting")


class TestNoMoreEvents:
    def test_pass_at_end(self):
        response = _make_response(_simple_events())
        response._cursor = 1
        response.no_more_events()

    def test_fail_when_events_remain(self):
        response = _make_response(_simple_events())
        with pytest.raises(AssertionError, match="Expected no more events"):
            response.no_more_events()


class TestFullSequence:
    async def test_call_then_judge_then_no_more(self):
        response = _make_response(_tool_call_events())
        response.function_called("get_weather", arguments={"location": "Tokyo"})
        await response.judge()
        response.no_more_events()

    async def test_multiple_tool_calls(self):
        events = [
            FunctionCallEvent(name="get_weather", arguments={"location": "Tokyo"}),
            FunctionCallOutputEvent(name="get_weather", output={"temp": 70}),
            FunctionCallEvent(name="get_news", arguments={"topic": "tech"}),
            FunctionCallOutputEvent(name="get_news", output=["headline1"]),
            ChatMessageEvent(role="assistant", content="Here's the info."),
        ]
        response = _make_response(events)
        response.function_called("get_weather")
        response.function_called("get_news", arguments={"topic": "tech"})
        await response.judge()
        response.no_more_events()

    def test_explicit_output_check(self):
        events = [
            FunctionCallEvent(name="get_weather", arguments={"location": "Tokyo"}),
            FunctionCallOutputEvent(
                name="get_weather", output={"temp": 70, "condition": "sunny"}
            ),
            ChatMessageEvent(role="assistant", content="Sunny, 70F."),
        ]
        response = _make_response(events)
        response._advance_to_type(FunctionCallEvent, "FunctionCallEvent")
        response.function_output(output={"temp": 70, "condition": "sunny"})


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
            response.function_called("nonexistent_tool")

    def test_includes_event_list_in_error(self):
        response = _make_response(_tool_call_events())
        with pytest.raises(AssertionError, match="FunctionCallEvent"):
            response.function_called("nonexistent_tool")
