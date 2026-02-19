"""Testing framework for Vision-Agents.

Provides text-only testing of agents without requiring audio/video
infrastructure or edge connections.

Example::

    from vision_agents.plugins import gemini
    from vision_agents.testing import TestEval

    async def test_greeting():
        llm = gemini.LLM("gemini-2.5-flash-lite")
        judge_llm = gemini.LLM("gemini-2.5-flash-lite")

        async with TestEval(llm=llm, judge=judge_llm, instructions="Be friendly") as session:
            await session.user_says("Hello")
            await session.agent_responds(intent="Friendly greeting")
            session.no_more_events()
"""

from vision_agents.testing._events import (
    ChatMessageEvent,
    FunctionCallEvent,
    FunctionCallOutputEvent,
    RunEvent,
)
from vision_agents.testing._mock_tools import mock_tools
from vision_agents.testing._run_result import RunResult
from vision_agents.testing._session import TestEval

TestSession = TestEval  # backwards compat alias

__all__ = [
    "TestEval",
    "TestSession",
    "RunResult",
    "ChatMessageEvent",
    "FunctionCallEvent",
    "FunctionCallOutputEvent",
    "RunEvent",
    "mock_tools",
]
