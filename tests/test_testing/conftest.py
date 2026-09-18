import pytest

from vision_agents.testing import Simulation
from vision_agents.testing.pytest_plugin import simulate  # noqa: F401

from ._fakes import ScriptedJudge, ScriptedLLM, user_done, user_says


@pytest.fixture
def simulation() -> Simulation:
    return Simulation(
        user_llm=lambda: ScriptedLLM(
            [user_says("Hi, move my appointment"), user_done()]
        )
    )


@pytest.fixture
def simulation_agent():
    return lambda: ScriptedLLM(["Done, moved to Friday 10am."])


@pytest.fixture
def simulation_judge() -> ScriptedJudge:
    return ScriptedJudge()
