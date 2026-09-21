"""Testing framework for Vision-Agents.

Provides text testing of agents without audio/video infrastructure or edge
connections, and audio-path simulation over an in-process loopback edge.

Evals test decisions: send one message, assert on tool calls and judge the
reply. Simulations test outcomes: an LLM plays the user from a YAML scenario
over several turns and a judge scores the whole transcript.

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

Simulate a multi-turn conversation from a YAML scenario::

    async def test_reschedule():
        scenario = load_scenario("scenarios/reschedule.yaml")
        simulation = Simulation(user_llm=lambda: gemini.LLM(MODEL))
        result = await simulation.run(
            lambda: setup_llm(MODEL),
            scenario,
            LLMJudge(gemini.LLM(MODEL)),
            instructions=INSTRUCTIONS,
        )
        assert result.passed, result.summary()

A scenario file holds one scenario::

    name: reschedule-appointment
    mode: text
    persona:
      impatient: true
    context:
      name: Alice
      appointment: Tuesday 3pm
    goal: Move appointment to Friday morning.
    constraints:
      - Reject anything after 11am.
    success:
      - appointment_rescheduled
      - correct_time_confirmed
    variations: 1
    repeat: 1

``variations: N`` holds N conversations with reworded briefs (the first is
always the scenario as written). ``repeat: k`` runs each variation k times
and reports ``pass@k`` and ``pass^k``. A judge failure marks a trial invalid
rather than failed. LLMs keep chat history, so pass factories (``lambda:
gemini.LLM(...)``) whenever a scenario runs more than one conversation.

``mode: audio`` runs the same scenario out loud: the simulated user's lines
are synthesised by ``caller_tts`` and paced into the agent's audio input
over a ``LoopbackEdge``, and the agent's TTS is transcribed back by
``caller_stt``. The judge reads what was heard; each turn also keeps what
the agent meant to say and its voice-to-voice latency::

    def create_agent() -> Agent:
        return Agent(edge=LoopbackEdge(), llm=..., stt=..., tts=..., ...)

    simulation = Simulation(user_llm=lambda: gemini.LLM(MODEL))
    result = await simulation.run(create_agent, load_scenario("spoken.yaml"), judge)

The same scenarios run from the command line with
``vision-agents agent simulate scenarios/``, which prints a table and writes
``report.json`` / ``report.md``.

Pytest fixtures for evals (``test_session``, ``judge``) live in
``vision_agents.testing.fixtures``; the ``simulate`` fixture for scenarios
lives in ``vision_agents.testing.pytest_plugin``. Enable them with
``pytest_plugins = [...]`` in ``conftest.py``.

Key exports:
    TestSession: async context manager that wraps an LLM or Agent for testing.
    TestResponse: returned by ``simple_response()`` — carries events and assertions.
    Judge: protocol for evaluation strategies.
    JudgeVerdict: dataclass returned by judges; holds per-criterion verdicts.
    JudgeError: raised by a judge that could not produce a verdict.
    Criterion / CriterionVerdict: a named check and its result.
    LLMJudge: default judge backed by an LLM instance.
    SAY_DO_CONSISTENCY, STAYS_IN_SCOPE, CONCISE, RESPONDS_IN_USER_LANGUAGE:
        built-in criteria.
    RunEvent: union of ChatMessageEvent, FunctionCallEvent, FunctionCallOutputEvent.
    Scenario: validated scenario definition; ``load_scenario`` reads one from YAML.
    SimulatedUser: LLM that plays the user from a scenario brief.
    Simulation: runs scenarios against an agent or LLM and judges the outcome.
    SimulationResult: trials plus ``passed``, ``pass_rate``, ``pass_at_k``, ``pass_pow_k``.
    Trial: one conversation — transcript, tool calls, turns, latencies, verdicts.
    Turn: one user message and the agent's ``TestResponse``; in audio mode also
        ``intended_reply`` and ``voice_to_voice_ms``.
    LoopbackEdge: in-process ``EdgeTransport`` with no network, for audio mode
        and for exercising an ``Agent``'s lifecycle in tests.
    generate_variations: reword a scenario N times keeping every fact.
    pass_at_k, pass_pow_k: estimators used for repeat reporting.
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
    JudgeError,
    JudgeVerdict,
    LLMJudge,
)
from vision_agents.testing._loopback import LoopbackEdge, LoopbackMicrophone
from vision_agents.testing._run_result import TestResponse
from vision_agents.testing._scenario import Scenario, load_scenario
from vision_agents.testing._session import TestSession
from vision_agents.testing._simulated_user import SimulatedUser, SimulatedUserError
from vision_agents.testing._simulation import (
    Simulation,
    SimulationResult,
    Trial,
    Turn,
    pass_at_k,
    pass_pow_k,
    render_transcript,
)
from vision_agents.testing._utils import collect_simple_response
from vision_agents.testing._variations import generate_variations

__all__ = [
    "Judge",
    "JudgeError",
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
    "LoopbackEdge",
    "LoopbackMicrophone",
    "ChatMessageEvent",
    "FunctionCallEvent",
    "FunctionCallOutputEvent",
    "RunEvent",
    "Scenario",
    "load_scenario",
    "SimulatedUser",
    "SimulatedUserError",
    "Simulation",
    "SimulationResult",
    "Trial",
    "Turn",
    "generate_variations",
    "pass_at_k",
    "pass_pow_k",
    "render_transcript",
    "collect_simple_response",
]
