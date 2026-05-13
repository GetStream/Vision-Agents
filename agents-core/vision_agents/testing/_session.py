import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock

from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.llm.llm import LLM

from ._events import (
    ChatMessageEvent,
    FunctionCallEvent,
    FunctionCallOutputEvent,
    RunEvent,
)
from ._mock_tools import mock_functions as _mock_functions
from ._run_result import TestResponse
from ._utils import collect_simple_response


class TestSession:
    """Test evaluator for running LLMs in text-only mode.

    Manages the LLM session lifecycle and sends text input.
    Returns ``TestResponse`` objects that carry both the data and
    assertion methods.

    Args:
        llm: The LLM instance to use, with tools already registered.
        instructions: System instructions for the agent.
    """

    __test__ = False

    def __init__(
        self,
        llm: LLM,
        instructions: str = "You are a helpful assistant.",
    ) -> None:
        self._llm = llm
        self._instructions = instructions
        self._conversation: InMemoryConversation | None = None
        self._captured_events: list[RunEvent] = []
        self._started = False

    async def __aenter__(self) -> "TestSession":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def start(self) -> None:
        """Initialize the session for testing."""
        if self._started:
            return

        self._llm.set_instructions(self._instructions)
        self._conversation = InMemoryConversation(
            instructions=self._instructions,
            messages=[],
        )
        self._llm.set_conversation(self._conversation)

        self._started = True

    async def close(self) -> None:
        """Clean up resources."""
        if not self._started:
            return

        self._started = False

    @property
    def llm(self) -> LLM:
        """The LLM instance (useful for ``mock_functions(session.llm, {...})``)."""
        return self._llm

    @contextmanager
    def mock_functions(
        self,
        mocks: dict[str, Callable[..., Any]],
    ) -> Generator[dict[str, AsyncMock], None, None]:
        """Temporarily replace tool implementations with ``AsyncMock`` wrappers.

        Thin wrapper around ``mock_functions(self._llm, mocks)``.

        Args:
            mocks: Mapping of tool name to mock callable.

        Yields:
            ``dict[str, AsyncMock]`` keyed by tool name.
        """
        with _mock_functions(self._llm, mocks) as wrapped:
            yield wrapped

    async def simple_response(self, text: str) -> TestResponse:
        """Send user text to the LLM and capture the response events.

        Conversation history accumulates across successive calls.

        Args:
            text: Text input simulating what a user would say.

        Returns:
            ``TestResponse`` with output, events, function_calls,
            timing, and assertion methods.
        """
        __tracebackhide__ = True
        if not self._started:
            raise RuntimeError(
                "TestSession not started. Use 'async with' or call start()."
            )

        start_time = time.monotonic()

        self._captured_events.clear()

        with self._observe_tool_calls():
            if self._conversation is not None:
                await self._conversation.send_message(
                    role="user",
                    user_id="test-user",
                    content=text,
                )

            _, response = await collect_simple_response(
                self._llm.simple_response(text=text)
            )

        events: list[RunEvent] = list(self._captured_events)
        if response.text:
            events.append(ChatMessageEvent(role="assistant", content=response.text))

            if self._conversation is not None:
                await self._conversation.send_message(
                    role="assistant",
                    user_id="test-agent",
                    content=response.text,
                )

        return TestResponse.build(
            events=events,
            user_input=text,
            start_time=start_time,
        )

    @contextmanager
    def _observe_tool_calls(self) -> Generator[None, None, None]:
        """Wrap registered tools so invocations are recorded into ``_captured_events``.

        Sits on top of any active ``mock_functions`` wrapper so mocks are
        observed too. Originals are restored on exit.
        """
        # TODO: not safe under overlapping simple_response() calls on the same
        # LLM — the second wrapper wraps the first and restoration depends on
        # exit order. Acceptable for now since tests run sessions sequentially.
        registry = self._llm.function_registry
        originals: dict[str, Callable[..., Any]] = {}
        for tool_name, fd in registry.functions.items():
            originals[tool_name] = fd.function
            fd.function = self._make_observer(tool_name, fd.function)
        try:
            yield
        finally:
            for tool_name, original in originals.items():
                registry.functions[tool_name].function = original

    def _make_observer(
        self, name: str, original: Callable[..., Any]
    ) -> Callable[..., Any]:
        """Build an async wrapper that records each call into ``_captured_events``."""

        # TODO: tool_call_id is not available at the registry call site, so
        # parallel invocations of the same tool within one response cannot be
        # paired in the captured trace. Sequential tool calls pair by adjacency.
        async def _observed(**kwargs: Any) -> Any:
            start = time.perf_counter()
            self._captured_events.append(
                FunctionCallEvent(name=name, arguments=kwargs, tool_call_id=None)
            )
            try:
                result = await original(**kwargs)
            except Exception as exc:
                elapsed = (time.perf_counter() - start) * 1000
                self._captured_events.append(
                    FunctionCallOutputEvent(
                        name=name,
                        output={"error": str(exc)},
                        is_error=True,
                        tool_call_id=None,
                        execution_time_ms=elapsed,
                    )
                )
                raise

            elapsed = (time.perf_counter() - start) * 1000
            self._captured_events.append(
                FunctionCallOutputEvent(
                    name=name,
                    output=result,
                    is_error=False,
                    tool_call_id=None,
                    execution_time_ms=elapsed,
                )
            )
            return result

        return _observed
