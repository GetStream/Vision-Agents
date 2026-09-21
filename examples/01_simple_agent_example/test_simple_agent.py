"""Integration tests for the 01_simple_agent_example.

The ``test_session``, ``judge`` and ``simulate`` fixtures come from
``vision_agents.testing`` and are configured in ``conftest.py``. The
single-turn tests are evals: they check one decision. The scenario tests are
simulations: an LLM plays the user from a file in ``scenarios/`` and the
judge scores the whole conversation.

Run:
    cd examples/01_simple_agent_example
    uv run py.test -m integration
"""

import os
from pathlib import Path

import pytest

from simple_agent_example import INSTRUCTIONS, setup_llm

from vision_agents.core import Agent, User
from vision_agents.plugins import deepgram, elevenlabs, gemini
from vision_agents.testing import (
    CONCISE,
    RESPONDS_IN_USER_LANGUAGE,
    SAY_DO_CONSISTENCY,
    STAYS_IN_SCOPE,
    LLMJudge,
    LoopbackEdge,
    Simulation,
    TestSession,
    load_scenario,
)

MODEL = os.getenv("VISION_AGENTS_TEST_MODEL", "gemini-3-flash-preview")


@pytest.mark.integration
async def test_greeting(test_session: TestSession, judge: LLMJudge):
    """Agent gives a friendly, short greeting without calling tools."""
    response = await test_session.simple_response("Hey there!")
    response.assert_function_not_called()

    verdict = await judge.evaluate_conversation(
        test_session.transcript,
        ["Gives a friendly greeting", CONCISE],
        instructions=test_session.instructions,
    )
    assert verdict.success, verdict.reason


@pytest.mark.integration
async def test_weather_tool_call(test_session: TestSession, judge: LLMJudge):
    """Agent calls get_weather with the right location and reports back."""
    response = await test_session.simple_response("What's the weather like in Berlin?")
    response.assert_function_called("get_weather", arguments={"location": "Berlin"})

    verdict = await judge.evaluate_conversation(
        test_session.transcript,
        ["Reports the current weather for Berlin", SAY_DO_CONSISTENCY, STAYS_IN_SCOPE],
        instructions=test_session.instructions,
    )
    assert verdict.success, verdict.reason


@pytest.mark.integration
async def test_weather_tool_call_mocked(test_session: TestSession, judge: LLMJudge):
    """Agent reports the mocked get_weather result instead of making things up."""
    with test_session.mock_functions(
        {"get_weather": lambda **_: {"temp_f": 55, "condition": "rainy"}}
    ):
        response = await test_session.simple_response(
            "What's the weather like in Berlin?"
        )

    response.assert_function_called("get_weather", arguments={"location": "Berlin"})
    response.assert_function_output(
        "get_weather", output={"temp_f": 55, "condition": "rainy"}
    )

    verdict = await judge.evaluate_conversation(
        test_session.transcript,
        ["Reports rainy weather for Berlin", SAY_DO_CONSISTENCY],
    )
    assert verdict.success, verdict.reason


@pytest.mark.integration
async def test_multi_turn_in_users_language(test_session: TestSession, judge: LLMJudge):
    """Agent keeps calling the tool and answering in German across turns."""
    await test_session.simple_response("Wie ist das Wetter in Berlin?")
    response = await test_session.simple_response("Und in Hamburg?")

    response.assert_function_called("get_weather")
    assert len([e for e in test_session.transcript if e.type == "function_call"]) == 2

    verdict = await judge.evaluate_conversation(
        test_session.transcript,
        [RESPONDS_IN_USER_LANGUAGE, SAY_DO_CONSISTENCY, CONCISE],
        instructions=test_session.instructions,
    )
    assert verdict.success, verdict.reason


@pytest.mark.integration
async def test_scenario_weather_trip_planning(simulate):
    """Simulated traveller works out whether to pack a rain jacket for Berlin."""
    result = await simulate(
        "scenarios/weather-trip-planning.yaml", instructions=INSTRUCTIONS
    )
    assert result.passed, result.summary()
    assert any(call.name == "get_weather" for call in result.trials[0].tool_calls)


@pytest.mark.integration
async def test_scenario_small_talk_then_weather(simulate):
    """Simulated user chats first, then asks for the weather in Tokyo."""
    result = await simulate(
        "scenarios/small-talk-then-weather.yaml", instructions=INSTRUCTIONS
    )
    assert result.passed, result.summary()
    assert result.trials[0].turn_count >= 2


@pytest.mark.integration
async def test_scenario_spoken_weather():
    """Simulated caller asks for Berlin's weather out loud over a loopback edge.

    The caller's line is spoken through ElevenLabs into the agent's Deepgram
    STT, and the agent's ElevenLabs reply is transcribed back by Deepgram;
    the judge reads that transcript, not the LLM's text.
    """
    for key in ("DEEPGRAM_API_KEY", "ELEVENLABS_API_KEY"):
        if not os.getenv(key):
            pytest.skip(f"{key} not set")

    def create_loopback_agent() -> Agent:
        return Agent(
            edge=LoopbackEdge(),
            agent_user=User(name="My happy AI friend", id="agent"),
            instructions=INSTRUCTIONS,
            llm=setup_llm(MODEL),
            tts=elevenlabs.TTS(model_id="eleven_flash_v2_5"),
            stt=deepgram.STT(eager_turn_detection=True),
        )

    scenario = load_scenario(
        Path(__file__).parent / "scenarios" / "spoken-weather.yaml"
    )
    simulation = Simulation(user_llm=lambda: gemini.LLM(MODEL), max_turns=4)
    result = await simulation.run(
        create_loopback_agent, scenario, LLMJudge(gemini.LLM(MODEL))
    )

    assert result.passed, result.summary()
    trial = result.trials[0]
    assert any(call.name == "get_weather" for call in trial.tool_calls)
    assert all(ms is not None for ms in trial.voice_to_voice_ms), trial.summary()
    assert all(turn.intended_reply for turn in trial.turns), trial.summary()
