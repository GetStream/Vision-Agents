"""Tool mocking for testing.

Temporarily replace tool implementations in a ``FunctionRegistry``
without changing the tool schema visible to the LLM.

Example::

    with mock_tools(llm, {"get_weather": lambda location: "sunny, 70F"}):
        result = await session.run("What's the weather?")
"""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

from vision_agents.core.llm.llm import LLM


@contextmanager
def mock_tools(
    llm: LLM,
    mocks: dict[str, Callable[..., Any]],
) -> Generator[None, None, None]:
    """Temporarily replace tool implementations for testing.

    The tool schemas (name, description, parameters) remain unchanged —
    only the underlying callable is swapped.  This lets you control return
    values or simulate errors while the LLM still "sees" the original tool
    definition.

    Args:
        llm: The LLM instance whose tools to mock.
        mocks: Mapping of tool name → mock callable.  The mock receives
            the same arguments as the original tool.  If the mock
            *returns* an ``Exception`` instance the framework will raise
            it during tool execution.

    Raises:
        KeyError: If a tool name is not registered on the LLM.
    """
    registry = llm.function_registry
    originals: dict[str, Callable[..., Any]] = {}

    for tool_name, mock_fn in mocks.items():
        func_def = registry._functions.get(tool_name)
        if func_def is None:
            raise KeyError(f"Tool '{tool_name}' is not registered on this LLM")
        originals[tool_name] = func_def.function
        func_def.function = mock_fn

    try:
        yield
    finally:
        for tool_name, original_fn in originals.items():
            func_def = registry._functions.get(tool_name)
            if func_def is not None:
                func_def.function = original_fn
