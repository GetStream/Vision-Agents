"""Tests for the pytest fixtures shipped in ``vision_agents.testing.fixtures``."""

import pytest

from tests.test_testing.fake_llms import ScriptedLLM, ToolCallingLLM
from vision_agents.core.llm.llm import LLM
from vision_agents.testing import ChatMessageEvent, LLMJudge, TestSession


@pytest.fixture
def agent_llm() -> LLM:
    return ToolCallingLLM(reply="hello")


@pytest.fixture
def agent_instructions() -> str:
    return "Only say hello."


@pytest.fixture
def judge_llm() -> LLM:
    return ScriptedLLM(
        '{"results": [{"name": "greets", "verdict": "pass", "score": 1.0, "reason": "ok"}]}'
    )


class TestFixtures:
    async def test_test_session_is_started_with_overrides(
        self, test_session: TestSession
    ):
        assert test_session.instructions == "Only say hello."

        response = await test_session.simple_response("hi")

        assert response.output == "hello"

    async def test_judge_wraps_judge_llm(self, judge: LLMJudge):
        verdict = await judge.evaluate_conversation(
            [
                ChatMessageEvent(role="user", content="hi"),
                ChatMessageEvent(role="assistant", content="hello"),
            ],
            ["greets"],
        )

        assert verdict.success is True
