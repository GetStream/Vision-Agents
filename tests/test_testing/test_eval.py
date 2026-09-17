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
        with pytest.raises(AssertionError, match="no matching call was found"):
            response.assert_function_called("wrong_tool")

    def test_argument_mismatch_raises(self):
        response = _make_response(_tool_call_events())
        with pytest.raises(AssertionError, match="no matching call was found"):
            response.assert_function_called(
                "get_weather", arguments={"location": "Berlin"}
            )

    def test_missing_argument_key_raises(self):
        response = _make_response(_tool_call_events())
        with pytest.raises(AssertionError, match="no matching call was found"):
            response.assert_function_called("get_weather", arguments={"city": None})

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
        with pytest.raises(AssertionError, match="no matching call was found"):
            response.assert_function_called("anything")

    def test_none_name_skips_name_check(self):
        response = _make_response(_tool_call_events())
        response.assert_function_called()
        assert response.function_calls[0].name == "get_weather"

    def test_matches_among_multiple_calls(self):
        events = [
            FunctionCallEvent(name="get_weather", arguments={"location": "Berlin"}),
            FunctionCallEvent(name="get_weather", arguments={"location": "Chicago"}),
            FunctionCallEvent(name="get_weather", arguments={"location": "Tokyo"}),
        ]
        response = _make_response(events)
        response.assert_function_called(
            "get_weather", arguments={"location": "Chicago"}
        )
        response.assert_function_called("get_weather", arguments={"location": "Tokyo"})
        response.assert_function_called("get_weather", arguments={"location": "Berlin"})


class TestFunctionOutput:
    def test_matches_output(self):
        events = [
            FunctionCallOutputEvent(
                name="get_weather", output={"temp": 70, "condition": "sunny"}
            ),
        ]
        response = _make_response(events)
        response.assert_function_output(
            "get_weather", output={"temp": 70, "condition": "sunny"}
        )

    def test_output_mismatch_raises(self):
        events = [
            FunctionCallOutputEvent(name="tool", output="actual"),
        ]
        response = _make_response(events)
        with pytest.raises(AssertionError, match="no matching output was found"):
            response.assert_function_output("tool", output="wrong")

    def test_is_error_match(self):
        events = [
            FunctionCallOutputEvent(
                name="tool", output={"error": "boom"}, is_error=True
            ),
        ]
        response = _make_response(events)
        response.assert_function_output("tool", is_error=True)

    def test_is_error_mismatch_raises(self):
        events = [
            FunctionCallOutputEvent(name="tool", output="ok", is_error=False),
        ]
        response = _make_response(events)
        with pytest.raises(AssertionError, match="no matching output was found"):
            response.assert_function_output("tool", is_error=True)

    def test_matches_among_multiple_outputs(self):
        events = [
            FunctionCallOutputEvent(name="get_weather", output={"temp": 70}),
            FunctionCallOutputEvent(name="get_weather", output={"temp": 55}),
            FunctionCallOutputEvent(name="get_weather", output={"temp": 30}),
        ]
        response = _make_response(events)
        response.assert_function_output("get_weather", output={"temp": 55})
        response.assert_function_output("get_weather", output={"temp": 30})
        response.assert_function_output("get_weather", output={"temp": 70})


class TestChatMessages:
    def test_finds_message(self):
        response = _make_response(_simple_events())
        assert len(response.chat_messages) == 1
        assert response.chat_messages[0].content == "Hello! How can I help?"

    def test_skips_non_message_events(self):
        response = _make_response(_tool_call_events())
        assert len(response.chat_messages) == 1
        assert (
            response.chat_messages[0].content == "The weather in Tokyo is sunny, 70F."
        )

    def test_no_message_returns_empty(self):
        events = [
            FunctionCallEvent(name="tool", arguments={}),
        ]
        response = _make_response(events)
        assert response.chat_messages == []


class TestFullSequence:
    def test_call_then_chat_message(self):
        response = _make_response(_tool_call_events())
        response.assert_function_called("get_weather", arguments={"location": "Tokyo"})
        assert len(response.chat_messages) == 1

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
        assert len(response.chat_messages) == 1

    def test_explicit_output_check(self):
        events = [
            FunctionCallEvent(name="get_weather", arguments={"location": "Tokyo"}),
            FunctionCallOutputEvent(
                name="get_weather", output={"temp": 70, "condition": "sunny"}
            ),
            ChatMessageEvent(role="assistant", content="Sunny, 70F."),
        ]
        response = _make_response(events)
        response.assert_function_output(
            "get_weather", output={"temp": 70, "condition": "sunny"}
        )


class TestTestResponse:
    def test_build_extracts_output(self):
        resp = _make_response(_tool_call_events())
        assert resp.input == "test input"
        assert resp.output == "The weather in Tokyo is sunny, 70F."
        assert len(resp.function_calls) == 1
        assert resp.function_calls[0].name == "get_weather"
        assert resp.duration_ms >= 0

    def test_build_no_chat_messages(self):
        events = [
            FunctionCallEvent(name="tool", arguments={}),
            FunctionCallOutputEvent(name="tool", output="ok"),
        ]
        resp = _make_response(events)
        assert resp.output is None
        assert len(resp.function_calls) == 1
        assert resp.chat_messages == []

    def test_build_simple_message(self):
        resp = _make_response(_simple_events())
        assert resp.output == "Hello! How can I help?"
        assert resp.function_calls == []


class TestErrorMessages:
    def test_includes_actual_calls_in_error(self):
        response = _make_response(_tool_call_events())
        with pytest.raises(AssertionError, match="Function calls:"):
            response.assert_function_called("nonexistent_tool")

    def test_includes_event_details_in_error(self):
        response = _make_response(_tool_call_events())
        with pytest.raises(
            AssertionError, match=r"FunctionCallEvent\(name='get_weather'"
        ):
            response.assert_function_called("nonexistent_tool")


class TestFunctionNotCalled:
    def test_passes_when_nothing_called(self):
        response = _make_response(_simple_events())
        response.assert_function_not_called()

    def test_passes_when_other_tool_called(self):
        response = _make_response(_tool_call_events())
        response.assert_function_not_called("send_email")

    def test_passes_when_arguments_differ(self):
        response = _make_response(_tool_call_events())
        response.assert_function_not_called(
            "get_weather", arguments={"location": "Berlin"}
        )

    def test_fails_when_called(self):
        response = _make_response(_tool_call_events())
        with pytest.raises(
            AssertionError, match="Expected no call to 'get_weather', but found 1"
        ):
            response.assert_function_not_called("get_weather")

    def test_fails_when_called_with_matching_arguments(self):
        response = _make_response(_tool_call_events())
        with pytest.raises(
            AssertionError,
            match=r"Expected no call to 'get_weather\(location='Tokyo'\)', but found 1",
        ):
            response.assert_function_not_called(
                "get_weather", arguments={"location": "Tokyo"}
            )

    def test_fails_when_any_function_called(self):
        response = _make_response(_tool_call_events())
        with pytest.raises(
            AssertionError, match="Expected no function call, but found 1"
        ):
            response.assert_function_not_called()

    def test_error_lists_actual_calls(self):
        response = _make_response(_tool_call_events())
        with pytest.raises(
            AssertionError,
            match=r"Function calls:\n   FunctionCallEvent\(name='get_weather'",
        ):
            response.assert_function_not_called("get_weather")


class TestFunctionCallOrder:
    @pytest.fixture
    def response(self) -> TestResponse:
        return _make_response(
            [
                FunctionCallEvent(name="search", arguments={"q": "pizza"}),
                FunctionCallOutputEvent(name="search", output=["Luigi's"]),
                FunctionCallEvent(name="book", arguments={"place": "Luigi's"}),
                FunctionCallOutputEvent(name="book", output="ok"),
                FunctionCallEvent(name="send_email", arguments={}),
                FunctionCallOutputEvent(name="send_email", output="sent"),
                ChatMessageEvent(role="assistant", content="Booked and emailed."),
            ]
        )

    def test_exact_order_passes(self, response: TestResponse):
        response.assert_function_call_order(["search", "book", "send_email"])

    def test_subsequence_passes(self, response: TestResponse):
        response.assert_function_call_order(["search", "send_email"])

    def test_single_name_passes(self, response: TestResponse):
        response.assert_function_call_order(["book"])

    def test_repeated_name_consumes_one_call_each(self):
        response = _make_response(
            [
                FunctionCallEvent(name="a", arguments={}),
                FunctionCallEvent(name="b", arguments={}),
                FunctionCallEvent(name="a", arguments={}),
            ]
        )
        response.assert_function_call_order(["a", "a"])
        response.assert_function_call_order(["a", "b", "a"])
        with pytest.raises(AssertionError, match="'a' was not called after 'a'"):
            response.assert_function_call_order(["a", "a", "a"])

    def test_wrong_order_fails(self, response: TestResponse):
        with pytest.raises(
            AssertionError,
            match=r"Expected function calls in order \['book', 'search'\], "
            r"but 'search' was not called after 'book'",
        ):
            response.assert_function_call_order(["book", "search"])

    def test_missing_name_fails(self, response: TestResponse):
        with pytest.raises(AssertionError, match="'pay' was not called after 'search'"):
            response.assert_function_call_order(["search", "pay"])

    def test_first_name_missing_fails_without_after(self, response: TestResponse):
        with pytest.raises(AssertionError, match=r"'pay' was not called\.\n"):
            response.assert_function_call_order(["pay"])

    def test_error_lists_actual_order_and_calls(self, response: TestResponse):
        with pytest.raises(
            AssertionError,
            match=r"Actual order: \['search', 'book', 'send_email'\]\n"
            r"Function calls:\n   FunctionCallEvent\(name='search'",
        ):
            response.assert_function_call_order(["send_email", "search"])

    def test_no_calls_fails(self):
        response = _make_response(_simple_events())
        with pytest.raises(AssertionError, match=r"Actual order: \[\]"):
            response.assert_function_call_order(["search"])

    def test_empty_names_raises(self, response: TestResponse):
        with pytest.raises(ValueError, match="at least one"):
            response.assert_function_call_order([])
