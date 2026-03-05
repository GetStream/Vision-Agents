"""Unit tests for TestSession.mock_functions."""

from unittest.mock import AsyncMock

import pytest

from vision_agents.core.edge.types import Participant
from vision_agents.core.llm.llm import LLM, LLMResponseEvent
from vision_agents.core.processors import Processor
from vision_agents.testing import TestSession


class _FakeLLM(LLM):
    """Minimal LLM that doesn't call a real model."""

    async def simple_response(
        self,
        text: str = "",
        processors: list[Processor] | None = None,
        participant: Participant | None = None,
    ) -> LLMResponseEvent:
        return LLMResponseEvent(original=None, text="fake")


async def _fake_weather(**_: object) -> dict:
    return {"temp": 70}


async def _fake_weather_99(**_: object) -> dict:
    return {"temp": 99}


async def _noop() -> None:
    return None


async def test_mock_functions_returns_async_mocks():
    llm = _FakeLLM()

    @llm.register_function(description="weather tool")
    async def get_weather(location: str) -> dict:
        return {"temp": 70}

    async with TestSession(llm=llm) as session:
        with session.mock_functions({"get_weather": _fake_weather}) as mocked:
            assert isinstance(mocked["get_weather"], AsyncMock)
            result = await mocked["get_weather"](location="Berlin")
            assert result == {"temp": 70}


async def test_mock_functions_assert_called():
    llm = _FakeLLM()

    @llm.register_function(description="weather tool")
    async def get_weather(location: str) -> dict:
        return {"temp": 70}

    async with TestSession(llm=llm) as session:
        with session.mock_functions({"get_weather": _fake_weather}) as mocked:
            await mocked["get_weather"](location="Berlin")
            mocked["get_weather"].assert_called_once()
            mocked["get_weather"].assert_called_with(location="Berlin")

            await mocked["get_weather"](location="Tokyo")
            assert mocked["get_weather"].call_count == 2


async def test_mock_functions_assert_not_called():
    llm = _FakeLLM()

    @llm.register_function(description="weather tool")
    async def get_weather(location: str) -> dict:
        return {"temp": 70}

    async with TestSession(llm=llm) as session:
        with session.mock_functions({"get_weather": _fake_weather}) as mocked:
            mocked["get_weather"].assert_not_called()


async def test_mock_functions_restores_on_exit():
    llm = _FakeLLM()

    @llm.register_function(description="weather tool")
    async def get_weather(location: str) -> dict:
        return {"temp": 70}

    original_fn = llm.function_registry._functions["get_weather"].function

    async with TestSession(llm=llm) as session:
        with session.mock_functions({"get_weather": _fake_weather_99}) as mocked:
            active_fn = llm.function_registry._functions["get_weather"].function
            assert active_fn is mocked["get_weather"]

        restored_fn = llm.function_registry._functions["get_weather"].function
        assert restored_fn is original_fn


async def test_mock_functions_unknown_tool_raises():
    llm = _FakeLLM()

    async with TestSession(llm=llm) as session:
        with pytest.raises(KeyError, match="nonexistent"):
            with session.mock_functions({"nonexistent": _noop}):
                pass
