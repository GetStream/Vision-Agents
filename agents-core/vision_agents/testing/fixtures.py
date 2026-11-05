"""Pytest fixtures for testing agents with ``vision_agents.testing``.

Enable them in your ``conftest.py``::

    pytest_plugins = ["vision_agents.testing.fixtures"]

Then override ``agent_llm`` (and optionally ``agent_instructions``) and
``judge_llm`` for your project::

    @pytest.fixture
    def agent_llm():
        return setup_llm()

    @pytest.fixture
    def judge_llm():
        return gemini.LLM("gemini-3-flash-preview")

Tests receive a started ``test_session`` and a ``judge``::

    async def test_greeting(test_session, judge):
        await test_session.simple_response("Hi!")
        verdict = await judge.evaluate_conversation(
            test_session.transcript, [CONCISE]
        )
        assert verdict.success, verdict.reason
"""

from collections.abc import AsyncIterator

import pytest

from vision_agents.core.llm.llm import LLM

from ._judge import LLMJudge
from ._session import TestSession


@pytest.fixture
def agent_llm() -> LLM:
    """The LLM under test. Override this fixture in your ``conftest.py``."""
    pytest.fail("Override the 'agent_llm' fixture to return the LLM under test.")


@pytest.fixture
def agent_instructions() -> str:
    """Instructions for the LLM under test. Override to match your agent."""
    return "You are a helpful assistant."


@pytest.fixture
async def test_session(
    agent_llm: LLM, agent_instructions: str
) -> AsyncIterator[TestSession]:
    """A started ``TestSession`` wrapping ``agent_llm``."""
    async with TestSession(llm=agent_llm, instructions=agent_instructions) as session:
        yield session


@pytest.fixture
def judge_llm() -> LLM:
    """The LLM that powers the judge. Override this fixture in your ``conftest.py``."""
    pytest.fail("Override the 'judge_llm' fixture to return the LLM used for judging.")


@pytest.fixture
def judge(judge_llm: LLM) -> LLMJudge:
    """An ``LLMJudge`` backed by ``judge_llm``."""
    return LLMJudge(judge_llm)
