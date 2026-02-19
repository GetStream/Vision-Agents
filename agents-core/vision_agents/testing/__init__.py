"""Testing framework for Vision-Agents.

Provides text-only testing of agents without requiring audio/video
infrastructure or edge connections.

Example::

    from vision_agents.plugins import gemini
    from vision_agents.testing import TestSession

    async def test_greeting():
        llm = gemini.LLM("gemini-2.5-flash-lite")
        judge_llm = gemini.LLM("gemini-2.5-flash-lite")

        async with TestSession(llm=llm, instructions="Be friendly") as session:
            result = await session.run("Hello")
            await (
                result.expect.next_event()
                .is_message(role="assistant")
                .judge(judge_llm, intent="Friendly greeting")
            )
            result.expect.no_more_events()
"""

from vision_agents.testing._events import (
    ChatMessageEvent,
    FunctionCallEvent,
    FunctionCallOutputEvent,
    RunEvent,
)
from vision_agents.testing._mock_tools import mock_tools
from vision_agents.testing._run_result import (
    ChatMessageAssert,
    EventAssert,
    EventRangeAssert,
    FunctionCallAssert,
    FunctionCallOutputAssert,
    RunAssert,
    RunResult,
)
from vision_agents.testing._session import TestSession

__all__ = [
    "TestSession",
    "RunResult",
    "RunAssert",
    "EventAssert",
    "EventRangeAssert",
    "ChatMessageAssert",
    "FunctionCallAssert",
    "FunctionCallOutputAssert",
    "ChatMessageEvent",
    "FunctionCallEvent",
    "FunctionCallOutputEvent",
    "RunEvent",
    "mock_tools",
]
