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
from vision_agents.core.llm.tool_call import context_parameter


@contextmanager
def mock_functions(
    llm: LLM,
    mocks: dict[str, Callable[..., Any]],
) -> Generator[dict[str, AsyncMock], None, None]:
    """Temporarily replace tool implementations with ``AsyncMock`` wrappers.

    Each mock callable is wrapped in an ``AsyncMock(side_effect=callable)``
    so that the original behaviour is preserved while call tracking
    (``assert_called_once``, ``assert_called_with``, etc.) becomes
    available via the standard ``unittest.mock`` API. A mock whose first
    parameter is annotated ``ToolCall`` is handed the call, whatever the tool
    it stands in for takes.

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
        if registry.functions.get(tool_name) is None:
            raise KeyError(f"Tool '{tool_name}' is not registered on this LLM")

    originals: dict[str, tuple[Callable[..., Any], bool]] = {}
    wrapped: dict[str, AsyncMock] = {}

    for tool_name, user_fn in mocks.items():
        func_def = registry.functions[tool_name]
        originals[tool_name] = (func_def.function, func_def.takes_context)
        async_mock = AsyncMock(side_effect=user_fn)
        wrapped[tool_name] = async_mock
        func_def.function = async_mock
        func_def.takes_context = context_parameter(user_fn) is not None

    try:
        yield wrapped
    finally:
        for tool_name, (original_fn, takes_context) in originals.items():
            registry.functions[tool_name].function = original_fn
            registry.functions[tool_name].takes_context = takes_context
