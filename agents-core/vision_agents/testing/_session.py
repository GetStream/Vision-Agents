import contextvars
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.llm.events import ToolEndEvent, ToolStartEvent
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

if TYPE_CHECKING:
    from vision_agents.core.agents import Agent

_DEFAULT_INSTRUCTIONS = "You are a helpful assistant."

# Events for the turn running in the current task. Tool events are dispatched
# on tasks that inherit this context, so overlapping turns stay separate.
_current_turn: contextvars.ContextVar[list[RunEvent] | None] = contextvars.ContextVar(
    "vision_agents_testing_current_turn", default=None
)


class TestSession:
    """Test evaluator for running an LLM or an Agent in text-only mode.

    Wraps either a bare ``LLM`` or a fully configured ``Agent``. In agent
    mode the agent's instructions and MCP tools are used, matching what
    production runs. Returns ``TestResponse`` objects that carry both the
    data and assertion methods.

    Args:
        llm: The LLM instance to use, with tools already registered.
            Mutually exclusive with ``agent``.
        instructions: System instructions for the LLM. Defaults to a generic
            assistant prompt. Cannot be combined with ``agent``.
        agent: Agent whose LLM, instructions and MCP servers are exercised.
    """

    __test__ = False

    def __init__(
        self,
        llm: LLM | None = None,
        instructions: str | None = None,
        *,
        agent: "Agent | None" = None,
    ) -> None:
        if agent is not None:
            if llm is not None:
                raise ValueError("Provide either 'llm' or 'agent', not both.")
            if instructions is not None:
                raise ValueError(
                    "'instructions' cannot be combined with 'agent'; "
                    "the agent's instructions are used."
                )
            self._llm: LLM = agent.llm
            self._instructions = agent.instructions.full_reference
        elif llm is not None:
            self._llm = llm
            self._instructions = (
                instructions if instructions is not None else _DEFAULT_INSTRUCTIONS
            )
        else:
            raise ValueError("Provide either 'llm' or 'agent'.")

        self._agent = agent
        self._conversation: InMemoryConversation | None = None
        self._transcript: list[RunEvent] = []
        self._active_turns: list[list[RunEvent]] = []
        self._started = False

    async def __aenter__(self) -> "TestSession":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def start(self) -> None:
        """Initialize the session for testing.

        In agent mode this also connects the agent's MCP servers so their
        tools are registered on the LLM.
        """
        if self._started:
            return

        self._llm.set_instructions(self._instructions)
        self._conversation = InMemoryConversation(
            instructions=self._instructions,
            messages=[],
        )
        self._llm.set_conversation(self._conversation)

        if self._agent is not None and self._agent.mcp_manager is not None:
            await self._agent.mcp_manager.connect_all()

        self._llm.events.subscribe(self._on_tool_event)
        self._started = True

    async def close(self) -> None:
        """Clean up resources."""
        if not self._started:
            return

        try:
            self._llm.events.unsubscribe(self._on_tool_event)
            if self._agent is not None and self._agent.mcp_manager is not None:
                await self._agent.mcp_manager.disconnect_all()
        finally:
            self._started = False

    @property
    def llm(self) -> LLM:
        """The LLM instance (useful for ``mock_functions(session.llm, {...})``)."""
        return self._llm

    @property
    def instructions(self) -> str:
        """The instructions the LLM runs with."""
        return self._instructions

    @property
    def transcript(self) -> list[RunEvent]:
        """All events captured so far, across every ``simple_response`` call."""
        return list(self._transcript)

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

        Conversation history accumulates across successive calls. The
        returned events start with the user message, followed by tool calls
        and outputs in the order they happened, and end with the assistant
        message.

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
        turn: list[RunEvent] = [ChatMessageEvent(role="user", content=text)]

        self._active_turns.append(turn)
        token = _current_turn.set(turn)
        try:
            if self._conversation is not None:
                await self._conversation.send_message(
                    role="user",
                    user_id="test-user",
                    content=text,
                )

            _, response = await collect_simple_response(
                self._llm.simple_response(text=text)
            )
            # Tool events are dispatched on separate tasks; make sure they
            # have all been recorded before building the response.
            await self._llm.events.wait()
        finally:
            _current_turn.reset(token)
            self._active_turns.remove(turn)

        if response.text:
            turn.append(ChatMessageEvent(role="assistant", content=response.text))

            if self._conversation is not None:
                await self._conversation.send_message(
                    role="assistant",
                    user_id="test-agent",
                    content=response.text,
                )

        self._transcript.extend(turn)
        return TestResponse.build(
            events=turn,
            user_input=text,
            start_time=start_time,
        )

    async def _on_tool_event(self, event: ToolStartEvent | ToolEndEvent) -> None:
        """Record LLM tool events into the turn they belong to."""
        turn = _current_turn.get()
        if turn is None:
            # Tool calls dispatched from a background task (e.g. a realtime
            # receive loop) do not inherit the turn's context; attribute
            # them to the most recently started turn.
            if not self._active_turns:
                return
            turn = self._active_turns[-1]
        turn.append(_to_run_event(event))


@contextmanager
def observe_tool_calls(llm: LLM, sink: list[RunEvent]) -> Generator[None, None, None]:
    """Record ``llm``'s tool calls and their results into ``sink`` while active."""

    async def _record(event: ToolStartEvent | ToolEndEvent) -> None:
        sink.append(_to_run_event(event))

    llm.events.subscribe(_record)
    try:
        yield
    finally:
        llm.events.unsubscribe(_record)


def _to_run_event(event: ToolStartEvent | ToolEndEvent) -> RunEvent:
    if isinstance(event, ToolStartEvent):
        return FunctionCallEvent(
            name=event.tool_name,
            arguments=dict(event.arguments or {}),
            tool_call_id=event.tool_call_id,
        )
    if event.success:
        return FunctionCallOutputEvent(
            name=event.tool_name,
            output=event.result,
            is_error=False,
            tool_call_id=event.tool_call_id,
            execution_time_ms=event.execution_time_ms,
        )
    return FunctionCallOutputEvent(
        name=event.tool_name,
        output={"error": event.error},
        is_error=True,
        tool_call_id=event.tool_call_id,
        execution_time_ms=event.execution_time_ms,
    )
