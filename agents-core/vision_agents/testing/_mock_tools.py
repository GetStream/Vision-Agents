"""Tool mocking for testing.

Temporarily replace tool implementations in a ``FunctionRegistry``
without changing the tool schema visible to the LLM.

Example::

    with mock_functions(llm, {"get_weather": lambda location: {"temp": 70}}) as mocked:
        result = await session.run("What's the weather?")
        mocked["get_weather"].assert_called_once()
"""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock

from vision_agents.core.llm.llm import LLM


@contextmanager
def mock_functions(
    llm: LLM,
    mocks: dict[str, Callable[..., Any]],
) -> Generator[dict[str, AsyncMock], None, None]:
    """Temporarily replace tool implementations with ``AsyncMock`` wrappers.

    Each mock callable is wrapped in an ``AsyncMock(side_effect=callable)``
    so that the original behaviour is preserved while call tracking
    (``assert_called_once``, ``assert_called_with``, etc.) becomes
    available via the standard ``unittest.mock`` API.

    Args:
        llm: The LLM instance whose tools to mock.
        mocks: Mapping of tool name to mock callable.

    Yields:
        ``dict[str, AsyncMock]`` keyed by tool name.

    Raises:
        KeyError: If a tool name is not registered on the LLM.
    """
    registry = llm.function_registry

    for tool_name in mocks:
        if registry._functions.get(tool_name) is None:
            raise KeyError(f"Tool '{tool_name}' is not registered on this LLM")

    originals: dict[str, Callable[..., Any]] = {}
    wrapped: dict[str, AsyncMock] = {}

    for tool_name, user_fn in mocks.items():
        func_def = registry._functions[tool_name]
        originals[tool_name] = func_def.function
        async_mock = AsyncMock(side_effect=user_fn)
        wrapped[tool_name] = async_mock
        func_def.function = async_mock

    try:
        yield wrapped
    finally:
        for tool_name, original_fn in originals.items():
            registry._functions[tool_name].function = original_fn
