import os

import pytest
from dotenv import load_dotenv

from simple_agent_example import setup_llm

from vision_agents.plugins import gemini
from vision_agents.testing import LLMJudge, Simulation
from vision_agents.testing.pytest_plugin import simulate  # noqa: F401

load_dotenv()

MODEL = os.getenv("VISION_AGENTS_TEST_MODEL", "gemini-3-flash-preview")


@pytest.fixture
def simulation() -> Simulation:
    return Simulation(user_llm=lambda: gemini.LLM(MODEL), max_turns=8)


@pytest.fixture
def simulation_agent():
    return lambda: setup_llm(MODEL)


@pytest.fixture
def simulation_judge() -> LLMJudge:
    return LLMJudge(gemini.LLM(MODEL))
