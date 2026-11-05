"""Testing framework for Vision-Agents.

Provides text-only testing of agents without requiring audio/video
infrastructure or edge connections.

Usage:

Judge a conversation against built-in and ad-hoc criteria::

    async def test_weather():
        judge = LLMJudge(gemini.LLM(MODEL))
        async with TestSession(llm=llm, instructions="...") as session:
            response = await session.simple_response("Weather in Tokyo?")
            response.assert_function_called("get_weather", arguments={"location": "Tokyo"})
            verdict = await judge.evaluate_conversation(
                session.transcript,
                ["Reports the weather for Tokyo", SAY_DO_CONSISTENCY, CONCISE],
                instructions=session.instructions,
            )
            assert verdict.success, verdict.reason

Wrap a full ``Agent`` so its instructions and MCP tools are used::

    async with TestSession(agent=agent) as session:
        response = await session.simple_response("Look up order 42")
        response.assert_function_call_order(["mcp_0_find_order", "notify_user"])

Pytest fixtures (``test_session``, ``judge``) live in
``vision_agents.testing.fixtures``; enable them with
``pytest_plugins = ["vision_agents.testing.fixtures"]`` in ``conftest.py``.

Key exports:
    TestSession: async context manager that wraps an LLM or Agent for testing.
    TestResponse: returned by ``simple_response()`` — carries events and assertions.
    Judge: protocol for evaluation strategies.
    JudgeVerdict: dataclass returned by judges; holds per-criterion verdicts.
    Criterion / CriterionVerdict: a named check and its result.
    LLMJudge: default judge backed by an LLM instance.
    SAY_DO_CONSISTENCY, STAYS_IN_SCOPE, CONCISE, RESPONDS_IN_USER_LANGUAGE:
        built-in criteria.
    RunEvent: union of ChatMessageEvent, FunctionCallEvent, FunctionCallOutputEvent.
"""

from vision_agents.testing._events import (
    ChatMessageEvent,
    FunctionCallEvent,
    FunctionCallOutputEvent,
    RunEvent,
)
from vision_agents.testing._judge import (
    CONCISE,
    RESPONDS_IN_USER_LANGUAGE,
    SAY_DO_CONSISTENCY,
    STAYS_IN_SCOPE,
    Criterion,
    CriterionVerdict,
    Judge,
    JudgeVerdict,
    LLMJudge,
)
from vision_agents.testing._run_result import TestResponse
from vision_agents.testing._session import TestSession
from vision_agents.testing._utils import collect_simple_response

__all__ = [
    "Judge",
    "JudgeVerdict",
    "Criterion",
    "CriterionVerdict",
    "LLMJudge",
    "SAY_DO_CONSISTENCY",
    "STAYS_IN_SCOPE",
    "CONCISE",
    "RESPONDS_IN_USER_LANGUAGE",
    "TestSession",
    "TestResponse",
    "ChatMessageEvent",
    "FunctionCallEvent",
    "FunctionCallOutputEvent",
    "RunEvent",
    "collect_simple_response",
]
