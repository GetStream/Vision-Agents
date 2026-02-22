import time
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
from vision_agents.testing._run_result import TestResponse


class TestSession:
    """Test evaluator for running LLMs in text-only mode.

    Manages the LLM session lifecycle and sends text input.
    Returns ``TestResponse`` objects that carry both the data and
    assertion methods.

    Args:
        llm: The LLM instance to use, with tools already registered.
        instructions: System instructions for the agent.
        judge: Optional LLM instance for intent evaluation. Required
            if ``response.judge(intent=...)`` is used.
    """

    __test__ = False

    def __init__(
        self,
        llm: LLM,
        instructions: str = "You are a helpful assistant.",
        judge: LLM | None = None,
    ) -> None:
        self._llm = llm
        self._instructions = instructions
        self._judge_llm = judge

        self._event_manager: EventManager | None = None
        self._conversation: InMemoryConversation | None = None
        self._captured_events: list[RunEvent] = []
        self._capturing = False
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
    def llm(self) -> LLM:
        """The LLM instance (useful for ``mock_tools(session.llm, {...})``)."""
        return self._llm

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
            judge_llm=self._judge_llm,
        )

    async def _on_tool_start(self, event: ToolStartEvent) -> None:
        if self._capturing:
            self._captured_events.append(
                FunctionCallEvent(
                    name=event.tool_name,
                    arguments=event.arguments or {},
                    tool_call_id=event.tool_call_id,
                )
            )

    async def _on_tool_end(self, event: ToolEndEvent) -> None:
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
