import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from vision_agents.core.instructions import Instructions
from vision_agents.plugins import gemini
from vision_agents.testing import LLMJudge, Simulation
from vision_agents.testing.pytest_plugin import simulate  # noqa: F401

load_dotenv()

MODEL = os.getenv("VISION_AGENTS_TEST_MODEL", "gemini-3-flash-preview")
EXAMPLE_DIR = Path(__file__).parent


@pytest.fixture
def instructions() -> str:
    return Instructions("Read @instructions.md", base_dir=EXAMPLE_DIR).full_reference


@pytest.fixture
def simulation() -> Simulation:
    return Simulation(user_llm=lambda: gemini.LLM(MODEL), max_turns=6)


@pytest.fixture
def simulation_agent():
    return lambda: gemini.LLM(MODEL)


@pytest.fixture
def simulation_judge() -> LLMJudge:
    return LLMJudge(gemini.LLM(MODEL))
