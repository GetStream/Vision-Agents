"""TestSession — the primary interface for testing Vision-Agents agents.

Works directly with an LLM instance in text-only mode (no audio, video,
or edge connection required).  Captures tool call events and the final
assistant response, returning a ``RunResult`` for fluent assertions.

Example::

    from vision_agents.plugins import gemini
    from vision_agents.testing import TestSession

    async def test_greeting():
        llm = gemini.LLM("gemini-2.5-flash-lite")
        async with TestSession(llm=llm, instructions="Be friendly") as session:
            result = await session.run("Hello")
            result.expect.next_event().is_message(role="assistant")
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.events.manager import EventManager
from vision_agents.core.llm.events import ToolEndEvent, ToolStartEvent
from vision_agents.core.llm.llm import LLM

from vision_agents.testing._events import (
    ChatMessageEvent,
    FunctionCallEvent,
    FunctionCallOutputEvent,
    RunEvent,
)
from vision_agents.testing._run_result import RunResult

if TYPE_CHECKING:
    from vision_agents.core.agents.agents import Agent

logger = logging.getLogger(__name__)

_evals_verbose = bool(int(os.getenv("VISION_AGENTS_EVALS_VERBOSE", "0")))


class TestSession:
    __test__ = False

    """Test session for running agents in text-only mode.

    Creates a lightweight testing wrapper around an LLM instance.
    Sends text input, captures tool call events and the final response,
    and returns a ``RunResult`` for assertion.

    Args:
        llm: The LLM instance to use.  This is the real LLM that the
            agent would use, with tools already registered.
        instructions: System instructions for the agent.

    Two creation patterns are supported::

        # 1. Direct LLM (recommended for tests)
        async with TestSession(llm=my_llm, instructions="...") as session:
            ...

        # 2. From an existing Agent
        async with TestSession.from_agent(agent) as session:
            ...
    """

    def __init__(
        self,
        llm: LLM,
        instructions: str = "You are a helpful assistant.",
    ) -> None:
        self._llm = llm
        self._instructions = instructions
        self._agent: Agent | None = None

        # Resolved at start()
        self._event_manager: EventManager | None = None
        self._conversation: InMemoryConversation | None = None
        self._captured_events: list[RunEvent] = []
        self._capturing = False
        self._started = False

    @classmethod
    def from_agent(cls, agent: Agent) -> TestSession:
        """Create a test session from an existing ``Agent``.

        Extracts the LLM and instructions without requiring a call
        connection.  The LLM's registered tools are preserved.
        """
        instructions_text = (
            agent.instructions.full_reference
            if hasattr(agent.instructions, "full_reference")
            else str(agent.instructions)
        )
        session = cls(llm=agent.llm, instructions=instructions_text)
        session._agent = agent
        return session

    # -- lifecycle --------------------------------------------------------

    async def __aenter__(self) -> TestSession:
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def start(self) -> None:
        """Initialize the session for testing."""
        if self._started:
            return

        self._llm.set_instructions(self._instructions)

        # Determine which EventManager to use.
        # After Agent.__init__ the LLM's EventManager is merged into the
        # Agent's (and its own processing task is stopped).  In that case
        # we must subscribe on the Agent's manager.
        if self._agent is not None:
            self._event_manager = self._agent.events
        else:
            self._event_manager = self._llm.events

        # Set up in-memory conversation for multi-turn support.
        self._conversation = InMemoryConversation(
            instructions=self._instructions,
            messages=[],
        )
        self._llm.set_conversation(self._conversation)

        # Subscribe to tool events once.
        self._event_manager.subscribe(self._on_tool_start)
        self._event_manager.subscribe(self._on_tool_end)

        self._started = True

    async def close(self) -> None:
        """Clean up resources."""
        if not self._started:
            return

        if self._event_manager is not None:
            self._event_manager.unsubscribe(self._on_tool_start)
            self._event_manager.unsubscribe(self._on_tool_end)

        self._started = False

    # -- event handlers ---------------------------------------------------

    async def _on_tool_start(self, event: ToolStartEvent):
        if self._capturing:
            self._captured_events.append(
                FunctionCallEvent(
                    name=event.tool_name,
                    arguments=event.arguments or {},
                    tool_call_id=event.tool_call_id,
                )
            )

    async def _on_tool_end(self, event: ToolEndEvent):
        if self._capturing:
            self._captured_events.append(
                FunctionCallOutputEvent(
                    name=event.tool_name,
                    output=event.result if event.success else {"error": event.error},
                    is_error=not event.success,
                    tool_call_id=event.tool_call_id,
                    execution_time_ms=event.execution_time_ms,
                )
            )

    # -- run --------------------------------------------------------------

    async def run(self, user_input: str) -> RunResult:
        """Execute a single conversation turn.

        Sends *user_input* to the LLM, captures all tool-call events
        and the final assistant message, and returns a ``RunResult``
        for assertion.

        The conversation history accumulates across successive ``run()``
        calls, enabling multi-turn testing.

        Args:
            user_input: Text input simulating what a user would say.

        Returns:
            ``RunResult`` containing captured events in chronological order.
        """
        if not self._started:
            raise RuntimeError(
                "TestSession not started. Use 'async with' or call start()."
            )

        # Reset per-turn state.
        self._captured_events.clear()
        self._capturing = True

        # Record user message in conversation history.
        if self._conversation is not None:
            await self._conversation.send_message(
                role="user",
                user_id="test-user",
                content=user_input,
            )

        try:
            response = await self._llm.simple_response(text=user_input)

            # Give the EventManager time to process queued events.
            if self._event_manager is not None:
                await self._event_manager.wait(timeout=5.0)

        finally:
            self._capturing = False

        # Append the final assistant message.
        events: list[RunEvent] = list(self._captured_events)
        if response and response.text:
            events.append(
                ChatMessageEvent(role="assistant", content=response.text)
            )

            # Also persist in conversation history for multi-turn.
            if self._conversation is not None:
                await self._conversation.send_message(
                    role="assistant",
                    user_id="test-agent",
                    content=response.text,
                )

        if _evals_verbose:
            logger.info(
                "TestSession.run completed: %d event(s) captured",
                len(events),
            )

        return RunResult(events=events, user_input=user_input)
