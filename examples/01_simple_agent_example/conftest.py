import os

import pytest
from dotenv import load_dotenv

from simple_agent_example import INSTRUCTIONS, setup_llm

from vision_agents.core.llm import LLM
from vision_agents.plugins import gemini

pytest_plugins = ["vision_agents.testing.fixtures"]

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
