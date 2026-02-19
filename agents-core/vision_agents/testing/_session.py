"""TestEval — the primary interface for testing Vision-Agents.

Works directly with an LLM instance in text-only mode (no audio, video,
or edge connection required).  Captures tool call events and the final
assistant response for fluent, scenario-style assertions.

Example::

    from vision_agents.plugins import gemini
    from vision_agents.testing import TestEval

    async def test_greeting():
        llm = gemini.LLM("gemini-2.5-flash-lite")
        judge_llm = gemini.LLM("gemini-2.5-flash-lite")
        async with TestEval(llm=llm, judge=judge_llm, instructions="Be friendly") as session:
            await session.simple_response("Hello")
            await session.judge(intent="Friendly greeting")
            session.no_more_events()
"""

from __future__ import annotations

import logging
import os
from typing import Any

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
from vision_agents.testing._run_result import RunResult, _format_events

logger = logging.getLogger(__name__)

_evals_verbose = bool(int(os.getenv("VISION_AGENTS_EVALS_VERBOSE", "0")))

_NOT_GIVEN = object()


class TestEval:
    __test__ = False

    """Test evaluator for running LLMs in text-only mode.

    Sends text input, captures tool call events and the final response,
    and provides scenario-style assertion methods.

    Args:
        llm: The LLM instance to use, with tools already registered.
        instructions: System instructions for the agent.
        judge: Optional LLM instance for intent evaluation. Required
            if ``judge(intent=...)`` is used.
    """

    def __init__(
        self,
        llm: LLM,
        instructions: str = "You are a helpful assistant.",
        judge: LLM | None = None,
    ) -> None:
        self._llm = llm
        self._instructions = instructions
        self._judge_llm = judge

        self._events: list[RunEvent] = []
        self._cursor: int = 0

        self._event_manager: EventManager | None = None
        self._conversation: InMemoryConversation | None = None
        self._captured_events: list[RunEvent] = []
        self._capturing = False
        self._started = False

    async def __aenter__(self) -> TestEval:
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def start(self) -> None:
        """Initialize the session for testing."""
        if self._started:
            return

        self._llm.set_instructions(self._instructions)
        self._event_manager = self._llm.events

        self._conversation = InMemoryConversation(
            instructions=self._instructions,
            messages=[],
        )
        self._llm.set_conversation(self._conversation)

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

    @property
    def events(self) -> list[RunEvent]:
        """Current turn's event list."""
        return self._events

    @property
    def llm(self) -> LLM:
        """The LLM instance (useful for ``mock_tools(session.llm, {...})``)."""
        return self._llm

    async def simple_response(self, text: str) -> RunResult:
        """Send user text to the LLM and capture the response events.

        Resets the assertion cursor to 0 for the new turn's events.
        Conversation history accumulates across successive calls.

        Args:
            text: Text input simulating what a user would say.

        Returns:
            ``RunResult`` for raw event access if needed.
        """
        __tracebackhide__ = True
        if not self._started:
            raise RuntimeError(
                "TestEval not started. Use 'async with' or call start()."
            )

        self._captured_events.clear()
        self._capturing = True

        if self._conversation is not None:
            await self._conversation.send_message(
                role="user",
                user_id="test-user",
                content=text,
            )

        try:
            response = await self._llm.simple_response(text=text)

            if self._event_manager is not None:
                await self._event_manager.wait(timeout=5.0)

        finally:
            self._capturing = False

        events: list[RunEvent] = list(self._captured_events)
        if response and response.text:
            events.append(
                ChatMessageEvent(role="assistant", content=response.text)
            )

            if self._conversation is not None:
                await self._conversation.send_message(
                    role="assistant",
                    user_id="test-agent",
                    content=response.text,
                )

        self._events = events
        self._cursor = 0

        if _evals_verbose:
            events_str = "\n    ".join(_format_events(self._events))
            print(
                f"\n+ simple_response(\"{text}\")\n"
                f"  events:\n    {events_str}\n"
            )

        return RunResult(events=events, user_input=text)

    def agent_calls(
        self,
        name: str | None = None,
        *,
        arguments: dict[str, Any] | None = None,
    ) -> FunctionCallEvent:
        """Assert the next event is a ``FunctionCallEvent``.

        Advances the cursor to the next ``FunctionCallEvent``, checks
        name and arguments (partial match), and auto-skips the following
        ``FunctionCallOutputEvent`` if present.

        Args:
            name: Expected function name. ``None`` to skip the check.
            arguments: Expected arguments (partial match — only specified
                keys are checked).

        Returns:
            The matched ``FunctionCallEvent``.
        """
        __tracebackhide__ = True
        event = self._advance_to_type(FunctionCallEvent, "FunctionCallEvent")
        assert isinstance(event, FunctionCallEvent)

        if name is not None and event.name != name:
            self._raise_with_debug_info(
                f"Expected call name '{name}', got '{event.name}'"
            )

        if arguments is not None:
            for key, value in arguments.items():
                actual = event.arguments.get(key)
                if actual != value:
                    self._raise_with_debug_info(
                        f"For argument '{key}', expected {value!r}, got {actual!r}"
                    )

        if (
            self._cursor < len(self._events)
            and isinstance(self._events[self._cursor], FunctionCallOutputEvent)
        ):
            self._cursor += 1

        return event

    def agent_calls_output(
        self,
        *,
        output: Any = _NOT_GIVEN,
        is_error: bool | None = None,
    ) -> FunctionCallOutputEvent:
        """Assert the next event is a ``FunctionCallOutputEvent``.

        Use this when you need to inspect the tool output explicitly.
        Advances the cursor to the next ``FunctionCallOutputEvent``.

        Args:
            output: Expected output value (exact match). Omit to skip.
            is_error: Expected error flag. ``None`` to skip the check.

        Returns:
            The matched ``FunctionCallOutputEvent``.
        """
        __tracebackhide__ = True
        event = self._advance_to_type(FunctionCallOutputEvent, "FunctionCallOutputEvent")
        assert isinstance(event, FunctionCallOutputEvent)

        if output is not _NOT_GIVEN and event.output != output:
            self._raise_with_debug_info(
                f"Expected output {output!r}, got {event.output!r}"
            )

        if is_error is not None and event.is_error != is_error:
            self._raise_with_debug_info(
                f"Expected is_error={is_error}, got {event.is_error}"
            )

        return event

    async def judge(
        self,
        *,
        intent: str | None = None,
    ) -> ChatMessageEvent:
        """Assert the next event is a ``ChatMessageEvent`` and optionally judge it.

        Advances the cursor to the next ``ChatMessageEvent``. If *intent*
        is given and a judge LLM was provided, evaluates whether the
        message fulfils the intent.

        Args:
            intent: Description of what the message should accomplish.
                Requires a judge LLM to have been set.

        Returns:
            The matched ``ChatMessageEvent``.
        """
        __tracebackhide__ = True
        event = self._advance_to_type(ChatMessageEvent, "ChatMessageEvent")
        assert isinstance(event, ChatMessageEvent)

        if intent is not None:
            if self._judge_llm is None:
                raise ValueError(
                    "Cannot evaluate intent without a judge LLM. "
                    "Pass judge=<llm> to TestEval()."
                )

            from vision_agents.testing._judge import evaluate_intent

            success, reason = await evaluate_intent(
                llm=self._judge_llm,
                message_content=event.content,
                intent=intent,
            )

            if not success:
                self._raise_with_debug_info(
                    f"Judgment failed: {reason}"
                )
            elif _evals_verbose:
                preview = event.content[:30].replace("\n", "\\n")
                print(f"  judgment passed for `{preview}...`: `{reason}`")

        return event

    def no_more_events(self) -> None:
        """Assert that no further events remain after the cursor."""
        __tracebackhide__ = True
        if self._cursor < len(self._events):
            event = self._events[self._cursor]
            self._raise_with_debug_info(
                f"Expected no more events, but found: {type(event).__name__}"
            )

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

    def _advance_to_type(self, expected_type: type, type_name: str) -> RunEvent:
        """Advance cursor to the next event of *expected_type*, skipping others."""
        __tracebackhide__ = True
        while self._cursor < len(self._events):
            event = self._events[self._cursor]
            self._cursor += 1
            if isinstance(event, expected_type):
                return event

        self._raise_with_debug_info(
            f"Expected {type_name}, but no matching event found."
        )
        raise RuntimeError("unreachable")

    def _raise_with_debug_info(self, message: str) -> None:
        __tracebackhide__ = True
        marker = max(0, self._cursor - 1)
        events_str = "\n".join(_format_events(self._events, selected_index=marker))
        raise AssertionError(f"{message}\nContext:\n{events_str}")
