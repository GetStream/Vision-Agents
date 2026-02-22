"""Unit tests for mock_tools."""

import pytest

from vision_agents.core.edge.types import Participant
from vision_agents.core.llm.llm import LLM, LLMResponseEvent
from vision_agents.core.processors import Processor
from vision_agents.testing import mock_tools


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
