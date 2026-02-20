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
            response = await session.simple_response("Hello")
            await response.judge(intent="Friendly greeting")
            response.no_more_events()
"""

from vision_agents.testing._events import (
    ChatMessageEvent,
    FunctionCallEvent,
    FunctionCallOutputEvent,
    RunEvent,
)
from vision_agents.testing._mock_tools import mock_tools
from vision_agents.testing._run_result import TestResponse
from vision_agents.testing._session import TestEval

__all__ = [
    "TestEval",
    "TestResponse",
    "ChatMessageEvent",
    "FunctionCallEvent",
    "FunctionCallOutputEvent",
    "RunEvent",
    "mock_tools",
]
