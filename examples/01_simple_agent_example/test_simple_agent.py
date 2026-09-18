"""Integration tests for the 01_simple_agent_example.

Run:
    cd examples/01_simple_agent_example
    uv run py.test -m integration
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from simple_agent_example import INSTRUCTIONS, setup_llm

from vision_agents.core import Agent, User
from vision_agents.plugins import deepgram, elevenlabs, gemini
from vision_agents.testing import (
    LLMJudge,
    LoopbackEdge,
    Simulation,
    TestSession,
    load_scenario,
)

load_dotenv()

MODEL = os.getenv("VISION_AGENTS_TEST_MODEL", "gemini-3-flash-preview")


def _skip_if_no_key():
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not set")


@pytest.mark.integration
async def test_greeting():
    """Agent gives a friendly, short greeting."""
    _skip_if_no_key()

    llm = setup_llm(MODEL)
    judge = LLMJudge(gemini.LLM(MODEL))

    async with TestSession(llm=llm, instructions=INSTRUCTIONS) as session:
        response = await session.simple_response("Hey there!")
        assert response.function_calls == []
        verdict = await judge.evaluate(
            response.chat_messages[0], intent="Friendly, short greeting"
        )
        assert verdict.success, verdict.reason


@pytest.mark.integration
async def test_weather_tool_call():
    """Agent calls get_weather with the right location and reports back."""
    _skip_if_no_key()

    llm = setup_llm(MODEL)
    judge = LLMJudge(gemini.LLM(MODEL))

    async with TestSession(llm=llm, instructions=INSTRUCTIONS) as session:
        response = await session.simple_response("What's the weather like in Berlin?")
        response.assert_function_called("get_weather", arguments={"location": "Berlin"})
        verdict = await judge.evaluate(
            response.chat_messages[0], intent="Reports current weather for Berlin"
        )
        assert verdict.success, verdict.reason


@pytest.mark.integration
async def test_weather_tool_call_mocked():
    """Agent calls get_weather with mocked return value; verify via AsyncMock."""
    _skip_if_no_key()

    llm = setup_llm(MODEL)
    judge = LLMJudge(gemini.LLM(MODEL))

    async with TestSession(llm=llm, instructions=INSTRUCTIONS) as session:
        with session.mock_functions(
            {"get_weather": lambda **_: {"temp_f": 55, "condition": "rainy"}}
        ) as mocked:
            response = await session.simple_response(
                "What's the weather like in Berlin?"
            )
            mocked["get_weather"].assert_called_once()
            mocked["get_weather"].assert_called_with(location="Berlin")
            response.assert_function_output(
                "get_weather", output={"temp_f": 55, "condition": "rainy"}
            )

            verdict = await judge.evaluate(
                response.chat_messages[0], intent="Reports rainy weather for Berlin"
            )
            assert verdict.success, verdict.reason


@pytest.mark.integration
async def test_scenario_weather_trip_planning(simulate):
    """Simulated traveller works out whether to pack a rain jacket for Berlin."""
    _skip_if_no_key()

    result = await simulate(
        "scenarios/weather-trip-planning.yaml", instructions=INSTRUCTIONS
    )
    assert result.passed, result.summary()
    assert any(call.name == "get_weather" for call in result.trials[0].tool_calls)


@pytest.mark.integration
async def test_scenario_small_talk_then_weather(simulate):
    """Simulated user chats first, then asks for the weather in Tokyo."""
    _skip_if_no_key()

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
    _skip_if_no_key()
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
