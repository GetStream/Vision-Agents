"""Unit tests for TestEval assertion methods.

These tests verify the v2 assertion API without requiring a real LLM.
Events are pre-populated directly on TestEval instances.
"""

import pytest

from vision_agents.testing._events import (
    ChatMessageEvent,
    FunctionCallEvent,
    FunctionCallOutputEvent,
)
from vision_agents.testing._session import TestEval


def _make_eval(events: list) -> TestEval:
    """Create a TestEval with pre-populated events for unit testing."""
    # Use a lightweight mock LLM — we only need assertion methods, not real LLM calls.
    eval_ = object.__new__(TestEval)
    eval_._events = events
    eval_._cursor = 0
    eval_._judge_llm = None
    return eval_


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


class TestAgentCalls:
    def test_matches_name(self):
        session = _make_eval(_tool_call_events())
        event = session.agent_calls("get_weather")
        assert event.name == "get_weather"

    def test_matches_arguments_partial(self):
        events = [
            FunctionCallEvent(
                name="search",
                arguments={"query": "hello", "limit": 10, "offset": 0},
            ),
            FunctionCallOutputEvent(name="search", output="results"),
        ]
        session = _make_eval(events)
        event = session.agent_calls("search", arguments={"query": "hello"})
        assert event.arguments["limit"] == 10

    def test_name_mismatch_raises(self):
        session = _make_eval(_tool_call_events())
        with pytest.raises(AssertionError, match="Expected call name 'wrong_tool'"):
            session.agent_calls("wrong_tool")

    def test_argument_mismatch_raises(self):
        session = _make_eval(_tool_call_events())
        with pytest.raises(AssertionError, match="argument 'location'"):
            session.agent_calls("get_weather", arguments={"location": "Berlin"})

    def test_wrong_event_type_skips_to_match(self):
        events = [
            ChatMessageEvent(role="assistant", content="Thinking..."),
            FunctionCallEvent(name="search", arguments={}),
            FunctionCallOutputEvent(name="search", output="ok"),
        ]
        session = _make_eval(events)
        event = session.agent_calls("search")
        assert event.name == "search"

    def test_no_function_call_raises(self):
        session = _make_eval(_simple_events())
        with pytest.raises(AssertionError, match="Expected FunctionCallEvent"):
            session.agent_calls("anything")

    async def test_auto_skips_function_call_output(self):
        session = _make_eval(_tool_call_events())
        session.agent_calls("get_weather")
        # Cursor should have auto-advanced past FunctionCallOutputEvent
        # so next assertion should find the ChatMessageEvent
        event = await session.agent_responds()
        assert isinstance(event, ChatMessageEvent)

    def test_none_name_skips_name_check(self):
        session = _make_eval(_tool_call_events())
        event = session.agent_calls()
        assert event.name == "get_weather"


class TestAgentCallsOutput:
    def test_matches_output(self):
        events = [
            FunctionCallOutputEvent(
                name="get_weather", output={"temp": 70, "condition": "sunny"}
            ),
        ]
        session = _make_eval(events)
        event = session.agent_calls_output(
            output={"temp": 70, "condition": "sunny"}
        )
        assert event.name == "get_weather"

    def test_output_mismatch_raises(self):
        events = [
            FunctionCallOutputEvent(name="tool", output="actual"),
        ]
        session = _make_eval(events)
        with pytest.raises(AssertionError, match="Expected output"):
            session.agent_calls_output(output="wrong")

    def test_is_error_match(self):
        events = [
            FunctionCallOutputEvent(
                name="tool", output={"error": "boom"}, is_error=True
            ),
        ]
        session = _make_eval(events)
        event = session.agent_calls_output(is_error=True)
        assert event.is_error is True

    def test_is_error_mismatch_raises(self):
        events = [
            FunctionCallOutputEvent(name="tool", output="ok", is_error=False),
        ]
        session = _make_eval(events)
        with pytest.raises(AssertionError, match="Expected is_error=True"):
            session.agent_calls_output(is_error=True)


class TestAgentResponds:
    async def test_finds_message(self):
        session = _make_eval(_simple_events())
        event = await session.agent_responds()
        assert event.content == "Hello! How can I help?"

    async def test_skips_non_message_events(self):
        session = _make_eval(_tool_call_events())
        event = await session.agent_responds()
        assert event.content == "The weather in Tokyo is sunny, 70F."

    async def test_no_message_raises(self):
        events = [
            FunctionCallEvent(name="tool", arguments={}),
        ]
        session = _make_eval(events)
        with pytest.raises(AssertionError, match="Expected ChatMessageEvent"):
            await session.agent_responds()

    async def test_intent_without_judge_raises(self):
        session = _make_eval(_simple_events())
        with pytest.raises(ValueError, match="Cannot evaluate intent"):
            await session.agent_responds(intent="Friendly greeting")


class TestNoMoreEvents:
    def test_pass_at_end(self):
        session = _make_eval(_simple_events())
        session._cursor = 1
        session.no_more_events()

    def test_fail_when_events_remain(self):
        session = _make_eval(_simple_events())
        with pytest.raises(AssertionError, match="Expected no more events"):
            session.no_more_events()


class TestFullSequence:
    def test_call_then_respond_then_no_more(self):
        session = _make_eval(_tool_call_events())
        session.agent_calls("get_weather", arguments={"location": "Tokyo"})
        # auto-skipped FunctionCallOutputEvent
        event = session._advance_to_type(ChatMessageEvent, "ChatMessageEvent")
        assert isinstance(event, ChatMessageEvent)
        session.no_more_events()

    async def test_multiple_tool_calls(self):
        events = [
            FunctionCallEvent(name="get_weather", arguments={"location": "Tokyo"}),
            FunctionCallOutputEvent(name="get_weather", output={"temp": 70}),
            FunctionCallEvent(name="get_news", arguments={"topic": "tech"}),
            FunctionCallOutputEvent(name="get_news", output=["headline1"]),
            ChatMessageEvent(role="assistant", content="Here's the info."),
        ]
        session = _make_eval(events)
        session.agent_calls("get_weather")
        session.agent_calls("get_news", arguments={"topic": "tech"})
        await session.agent_responds()
        session.no_more_events()

    def test_explicit_output_check(self):
        events = [
            FunctionCallEvent(name="get_weather", arguments={"location": "Tokyo"}),
            FunctionCallOutputEvent(
                name="get_weather", output={"temp": 70, "condition": "sunny"}
            ),
            ChatMessageEvent(role="assistant", content="Sunny, 70F."),
        ]
        session = _make_eval(events)
        # Don't use agent_calls (which auto-skips output); manually advance
        session._advance_to_type(FunctionCallEvent, "FunctionCallEvent")
        session.agent_calls_output(output={"temp": 70, "condition": "sunny"})


class TestErrorMessages:
    def test_includes_context_in_error(self):
        session = _make_eval(_tool_call_events())
        with pytest.raises(AssertionError, match="Context:"):
            session.agent_calls("nonexistent_tool")

    def test_includes_event_list_in_error(self):
        session = _make_eval(_tool_call_events())
        with pytest.raises(AssertionError, match="FunctionCallEvent"):
            session.agent_calls("nonexistent_tool")
