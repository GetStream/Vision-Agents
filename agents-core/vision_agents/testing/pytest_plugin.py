"""Pytest fixtures for scenario simulations.

Register the plugin in your top-level ``conftest.py`` (or import the
``simulate`` fixture into any ``conftest.py``) and override the three
configuration fixtures for your agent::

    pytest_plugins = ["vision_agents.testing.pytest_plugin"]
    # or: from vision_agents.testing.pytest_plugin import simulate  # noqa: F401

    @pytest.fixture
    def simulation():
        return Simulation(user_llm=lambda: gemini.LLM(MODEL))

    @pytest.fixture
    def simulation_agent():
        return lambda: setup_llm(MODEL)

    @pytest.fixture
    def simulation_judge():
        return LLMJudge(gemini.LLM(MODEL))

A test is then a scenario path plus an assertion::

    async def test_reschedule(simulate):
        result = await simulate("scenarios/reschedule.yaml", instructions=INSTRUCTIONS)
        assert result.passed, result.summary()

Relative scenario paths resolve against the test file's directory.
"""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from ._judge import Judge
from ._scenario import Scenario, load_scenario
from ._simulation import Simulation, SimulationResult, Target, TargetFactory

Simulate = Callable[..., Awaitable[SimulationResult]]


@pytest.fixture
def simulation() -> Simulation:
    """Override to return a configured ``Simulation``."""
    raise NotImplementedError(
        "Override the 'simulation' fixture to return a Simulation instance"
    )


@pytest.fixture
def simulation_agent() -> Target | TargetFactory:
    """Override to return the agent or LLM under test, or a factory building one."""
    raise NotImplementedError(
        "Override the 'simulation_agent' fixture to return an Agent, an LLM or a factory"
    )


@pytest.fixture
def simulation_judge() -> Judge:
    """Override to return the judge used for success criteria."""
    raise NotImplementedError(
        "Override the 'simulation_judge' fixture to return a Judge"
    )


@pytest.fixture
def simulate(
    request: pytest.FixtureRequest,
    simulation: Simulation,
    simulation_agent: Target | TargetFactory,
    simulation_judge: Judge,
) -> Simulate:
    """Run a scenario file (or ``Scenario``) against the configured agent."""
    test_dir = request.path.parent

    async def _simulate(
        scenario: str | Path | Scenario, instructions: str | None = None
    ) -> SimulationResult:
        if not isinstance(scenario, Scenario):
            path = Path(scenario)
            if not path.is_absolute():
                path = test_dir / path
            scenario = await asyncio.to_thread(load_scenario, path)
        return await simulation.run(
            simulation_agent, scenario, simulation_judge, instructions=instructions
        )

    return _simulate
