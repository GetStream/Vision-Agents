import os
from collections.abc import Callable

import pytest
from dotenv import load_dotenv

from simple_agent_example import INSTRUCTIONS, setup_llm

from vision_agents.core.llm import LLM
from vision_agents.plugins import gemini
from vision_agents.testing import LLMJudge, Simulation
from vision_agents.testing.fixtures import judge, test_session  # noqa: F401
from vision_agents.testing.pytest_plugin import simulate  # noqa: F401

load_dotenv()

MODEL = os.getenv("VISION_AGENTS_TEST_MODEL", "gemini-3-flash-preview")


@pytest.fixture(autouse=True)
def _require_api_key() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not set")


@pytest.fixture
def agent_llm() -> LLM:
    return setup_llm(MODEL)


@pytest.fixture
def agent_instructions() -> str:
    return INSTRUCTIONS


@pytest.fixture
def judge_llm() -> LLM:
    return gemini.LLM(MODEL)


@pytest.fixture
def simulation() -> Simulation:
    return Simulation(user_llm=lambda: gemini.LLM(MODEL), max_turns=8)


@pytest.fixture
def simulation_agent() -> Callable[[], LLM]:
    return lambda: setup_llm(MODEL)


@pytest.fixture
def simulation_judge() -> LLMJudge:
    return LLMJudge(gemini.LLM(MODEL))
