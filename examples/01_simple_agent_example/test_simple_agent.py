"""Integration tests for the 01_simple_agent_example.

The ``test_session`` and ``judge`` fixtures come from
``vision_agents.testing.fixtures`` and are configured in ``conftest.py``.

Run:
    cd examples/01_simple_agent_example
    uv run py.test -m integration
"""

import pytest

from vision_agents.testing import (
    CONCISE,
    RESPONDS_IN_USER_LANGUAGE,
    SAY_DO_CONSISTENCY,
    STAYS_IN_SCOPE,
    LLMJudge,
    TestSession,
)


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
    with test_session.mock_functions(
        {"get_weather": lambda **_: {"temp_f": 60, "condition": "cloudy"}}
    ):
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
