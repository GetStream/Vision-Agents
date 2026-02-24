"""Unit tests for mock_tools and mock_functions."""

from unittest.mock import AsyncMock

import pytest

from vision_agents.core.edge.types import Participant
from vision_agents.core.llm.llm import LLM, LLMResponseEvent
from vision_agents.core.processors import Processor
from vision_agents.testing import mock_functions, mock_tools


class _FakeLLM(LLM):
    """Minimal LLM that doesn't call a real model."""

    async def simple_response(
        self,
        text: str = "",
        processors: list[Processor] | None = None,
        participant: Participant | None = None,
    ) -> LLMResponseEvent:
        return LLMResponseEvent(original=None, text="fake")


async def test_mock_tools_swaps_and_restores():
    llm = _FakeLLM()

    @llm.register_function(description="original tool")
    def my_tool(x: int) -> int:
        return x * 2

    original_fn = llm.function_registry._functions["my_tool"].function

    def mock_fn(x: int) -> int:
        return x * 100

    with mock_tools(llm, {"my_tool": mock_fn}):
        active_fn = llm.function_registry._functions["my_tool"].function
        assert active_fn is mock_fn
        assert llm.function_registry.call_function("my_tool", {"x": 5}) == 500

    restored_fn = llm.function_registry._functions["my_tool"].function
    assert restored_fn is original_fn
    assert llm.function_registry.call_function("my_tool", {"x": 5}) == 10


async def test_mock_tools_restores_on_exception():
    llm = _FakeLLM()

    @llm.register_function(description="tool")
    def my_tool(x: int) -> int:
        return x

    original_fn = llm.function_registry._functions["my_tool"].function

    with pytest.raises(ValueError, match="boom"):
        with mock_tools(llm, {"my_tool": lambda x: x}):
            raise ValueError("boom")

    assert llm.function_registry._functions["my_tool"].function is original_fn


async def test_mock_tools_unknown_tool():
    llm = _FakeLLM()

    with pytest.raises(KeyError, match="nonexistent"):
        with mock_tools(llm, {"nonexistent": lambda: None}):
            pass


async def test_mock_tools_multiple_tools():
    llm = _FakeLLM()

    @llm.register_function(description="a")
    def tool_a(x: int) -> int:
        return x

    @llm.register_function(description="b")
    def tool_b(y: str) -> str:
        return y

    with mock_tools(llm, {"tool_a": lambda x: 999, "tool_b": lambda y: "mocked"}):
        assert llm.function_registry.call_function("tool_a", {"x": 1}) == 999
        assert llm.function_registry.call_function("tool_b", {"y": "hi"}) == "mocked"

    assert llm.function_registry.call_function("tool_a", {"x": 1}) == 1
    assert llm.function_registry.call_function("tool_b", {"y": "hi"}) == "hi"


async def test_mock_tools_unknown_with_valid_does_not_mutate():
    """A KeyError for one tool must not leave other tools partially swapped."""
    llm = _FakeLLM()

    @llm.register_function(description="a")
    def tool_a(x: int) -> int:
        return x

    original_fn = llm.function_registry._functions["tool_a"].function

    with pytest.raises(KeyError, match="nonexistent"):
        with mock_tools(llm, {"tool_a": lambda x: 999, "nonexistent": lambda: None}):
            pass

    assert llm.function_registry._functions["tool_a"].function is original_fn


async def test_mock_functions_returns_async_mocks():
    llm = _FakeLLM()

    @llm.register_function(description="weather tool")
    def get_weather(location: str) -> dict:
        return {"temp": 70}

    with mock_functions(llm, {"get_weather": lambda **_: {"temp": 70}}) as mocked:
        assert isinstance(mocked["get_weather"], AsyncMock)
        result = await mocked["get_weather"](location="Berlin")
        assert result == {"temp": 70}


async def test_mock_functions_assert_called():
    llm = _FakeLLM()

    @llm.register_function(description="weather tool")
    def get_weather(location: str) -> dict:
        return {"temp": 70}

    with mock_functions(llm, {"get_weather": lambda **_: {"temp": 70}}) as mocked:
        await mocked["get_weather"](location="Berlin")
        mocked["get_weather"].assert_called_once()
        mocked["get_weather"].assert_called_with(location="Berlin")

        await mocked["get_weather"](location="Tokyo")
        assert mocked["get_weather"].call_count == 2


async def test_mock_functions_assert_not_called():
    llm = _FakeLLM()

    @llm.register_function(description="weather tool")
    def get_weather(location: str) -> dict:
        return {"temp": 70}

    with mock_functions(llm, {"get_weather": lambda **_: {"temp": 70}}) as mocked:
        mocked["get_weather"].assert_not_called()


async def test_mock_functions_restores_on_exit():
    llm = _FakeLLM()

    @llm.register_function(description="weather tool")
    def get_weather(location: str) -> dict:
        return {"temp": 70}

    original_fn = llm.function_registry._functions["get_weather"].function

    with mock_functions(llm, {"get_weather": lambda **_: {"temp": 99}}) as mocked:
        active_fn = llm.function_registry._functions["get_weather"].function
        assert active_fn is mocked["get_weather"]

    restored_fn = llm.function_registry._functions["get_weather"].function
    assert restored_fn is original_fn


async def test_mock_functions_unknown_tool_raises():
    llm = _FakeLLM()

    with pytest.raises(KeyError, match="nonexistent"):
        with mock_functions(llm, {"nonexistent": lambda: None}):
            pass
