"""Tests for LLMJudge and the built-in criteria."""

import json
import os
from typing import Any

import pytest

from tests.test_testing.fake_llms import ScriptedLLM
from vision_agents.plugins import gemini
from vision_agents.testing import (
    CONCISE,
    RESPONDS_IN_USER_LANGUAGE,
    SAY_DO_CONSISTENCY,
    STAYS_IN_SCOPE,
    ChatMessageEvent,
    Criterion,
    FunctionCallEvent,
    FunctionCallOutputEvent,
    LLMJudge,
    RunEvent,
)

MODEL = os.getenv("VISION_AGENTS_TEST_MODEL", "gemini-3-flash-preview")

requires_gemini = pytest.mark.skipif(
    not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
    reason="GOOGLE_API_KEY or GEMINI_API_KEY not set; skipping live integration test",
)


def _results(*items: dict[str, Any]) -> str:
    return json.dumps({"results": list(items)})


def _chat(*turns: tuple[str, str]) -> list[RunEvent]:
    return [ChatMessageEvent(role=role, content=content) for role, content in turns]


@pytest.fixture
def transcript() -> list[RunEvent]:
    return [
        ChatMessageEvent(role="user", content="What's the weather in Berlin?"),
        FunctionCallEvent(
            name="get_weather", arguments={"location": "Berlin"}, tool_call_id="call_1"
        ),
        FunctionCallOutputEvent(
            name="get_weather", output={"temp_f": 70}, tool_call_id="call_1"
        ),
        ChatMessageEvent(role="assistant", content="It is 70F in Berlin."),
    ]


_BOOKING_REQUEST = ("user", "Book a table for two at 7pm tonight.")
_BOOKING_CLAIM = (
    "assistant",
    "Done, your table for two at 7pm tonight is booked. Confirmation R123.",
)
_SAY_DO_PASS: list[RunEvent] = [
    ChatMessageEvent(*_BOOKING_REQUEST),
    FunctionCallEvent(
        name="book_table", arguments={"guests": 2, "time": "19:00"}, tool_call_id="c1"
    ),
    FunctionCallOutputEvent(
        name="book_table", output={"confirmation": "R123"}, tool_call_id="c1"
    ),
    ChatMessageEvent(*_BOOKING_CLAIM),
]
_SAY_DO_FAIL: list[RunEvent] = _chat(_BOOKING_REQUEST, _BOOKING_CLAIM)

_WEATHER_INSTRUCTIONS = "You are a weather assistant. Only answer weather questions."
_WEATHER_QUESTION = ("user", "What's the weather in Berlin?")
_SCOPE_PASS = _chat(_WEATHER_QUESTION, ("assistant", "It's 18C and cloudy in Berlin."))
_SCOPE_FAIL = _chat(
    _WEATHER_QUESTION,
    (
        "assistant",
        "It's 18C and cloudy in Berlin. By the way, here is my favourite lasagna "
        "recipe, and you should really consider buying tech stocks this week.",
    ),
)

_TIME_QUESTION = ("user", "What time is it in Tokyo?")
_CONCISE_PASS = _chat(_TIME_QUESTION, ("assistant", "It's 9pm in Tokyo."))
_CONCISE_FAIL = _chat(
    _TIME_QUESTION,
    (
        "assistant",
        "That's a great question! So you want to know what time it is in Tokyo. "
        "Let me think about that for a second. Tokyo is in Japan, and Japan uses "
        "Japan Standard Time, which does not observe daylight saving. Taking all "
        "of that into account, and I really hope this is helpful for you, the "
        "current time in Tokyo is 9pm. Please let me know if there is anything "
        "else at all I can help you with today!",
    ),
)

_SPANISH_QUESTION = ("user", "¿Qué tiempo hace en Madrid?")
_LANGUAGE_PASS = _chat(
    _SPANISH_QUESTION, ("assistant", "Hace sol y 25 grados en Madrid.")
)
_LANGUAGE_FAIL = _chat(
    _SPANISH_QUESTION, ("assistant", "It's sunny and 25 degrees in Madrid.")
)


class TestLLMJudge:
    async def test_returns_a_verdict_per_criterion(self, transcript):
        llm = ScriptedLLM(
            _results(
                {
                    "name": "reports_weather",
                    "verdict": "pass",
                    "score": 0.9,
                    "reason": "Reports 70F.",
                },
                {
                    "name": "concise",
                    "verdict": "fail",
                    "score": 0.3,
                    "reason": "Too wordy.",
                },
            )
        )
        judge = LLMJudge(llm)

        verdict = await judge.evaluate_conversation(
            transcript,
            [Criterion("reports_weather", "Reports the weather for Berlin"), CONCISE],
        )

        assert verdict.success is False
        assert verdict.score == pytest.approx(0.6)
        assert [c.name for c in verdict.criteria] == ["reports_weather", "concise"]
        assert verdict.criteria[0].success is True
        assert verdict.criteria[0].score == 0.9
        assert verdict.criteria[0].reason == "Reports 70F."
        assert verdict.criteria[1].success is False
        assert verdict.reason == "concise: Too wordy."

    async def test_all_criteria_passing_is_success(self, transcript):
        llm = ScriptedLLM(
            _results(
                {"name": "a", "verdict": "pass", "score": 1.0, "reason": "ok"},
                {"name": "b", "verdict": "PASS", "score": 0.8, "reason": "fine"},
            )
        )

        verdict = await LLMJudge(llm).evaluate_conversation(
            transcript, [Criterion("a", "A"), Criterion("b", "B")]
        )

        assert verdict.success is True
        assert verdict.score == pytest.approx(0.9)
        assert verdict.reason == "a: ok\nb: fine"

    async def test_string_criterion_uses_text_as_name(self, transcript):
        llm = ScriptedLLM(
            _results({"name": "Reports the weather", "verdict": "pass", "score": 1})
        )

        verdict = await LLMJudge(llm).evaluate_conversation(
            transcript, ["Reports the weather"]
        )

        assert verdict.success is True
        assert verdict.criteria[0].name == "Reports the weather"
        assert "- Reports the weather: Reports the weather" in llm.prompts[0]

    async def test_missing_result_fails_that_criterion(self, transcript):
        llm = ScriptedLLM(_results({"name": "a", "verdict": "pass", "score": 1.0}))

        verdict = await LLMJudge(llm).evaluate_conversation(
            transcript, [Criterion("a", "A"), Criterion("b", "B")]
        )

        assert verdict.success is False
        assert verdict.score == pytest.approx(0.5)
        assert verdict.criteria[1].reason == (
            "Judge returned no verdict for this criterion."
        )

    async def test_malformed_json_fails(self, transcript):
        verdict = await LLMJudge(ScriptedLLM("not json")).evaluate_conversation(
            transcript, [CONCISE]
        )

        assert verdict.success is False
        assert "Could not parse JSON" in verdict.reason
        assert verdict.criteria == []

    async def test_missing_results_list_fails(self, transcript):
        verdict = await LLMJudge(
            ScriptedLLM('{"verdict": "pass"}')
        ).evaluate_conversation(transcript, [CONCISE])

        assert verdict.success is False
        assert "Missing 'results'" in verdict.reason

    async def test_code_fence_is_stripped(self, transcript):
        llm = ScriptedLLM(
            "```json\n"
            + _results({"name": "concise", "verdict": "pass", "score": 1.0})
            + "\n```"
        )

        verdict = await LLMJudge(llm).evaluate_conversation(transcript, [CONCISE])

        assert verdict.success is True

    async def test_unknown_verdict_fails(self, transcript):
        llm = ScriptedLLM(_results({"name": "concise", "verdict": "maybe"}))

        verdict = await LLMJudge(llm).evaluate_conversation(transcript, [CONCISE])

        assert verdict.success is False
        assert verdict.criteria[0].score == 0.0
        assert "Unknown verdict 'maybe'" in verdict.criteria[0].reason

    async def test_score_is_clamped_and_defaults_to_the_verdict(self, transcript):
        llm = ScriptedLLM(
            _results(
                {"name": "a", "verdict": "pass", "score": 7},
                {"name": "b", "verdict": "pass"},
                {"name": "c", "verdict": "fail", "score": "high"},
            )
        )

        verdict = await LLMJudge(llm).evaluate_conversation(
            transcript, [Criterion("a", "A"), Criterion("b", "B"), Criterion("c", "C")]
        )

        assert [c.score for c in verdict.criteria] == [1.0, 1.0, 0.0]
        assert verdict.criteria[1].reason == "Passed."
        assert verdict.criteria[2].reason == "Failed."

    async def test_empty_conversation_fails_without_calling_llm(self):
        llm = ScriptedLLM("unused")

        verdict = await LLMJudge(llm).evaluate_conversation([], [CONCISE])

        assert verdict.success is False
        assert verdict.reason == "The conversation is empty."
        assert llm.prompts == []

    async def test_empty_criteria_fails_without_calling_llm(self, transcript):
        llm = ScriptedLLM("unused")

        verdict = await LLMJudge(llm).evaluate_conversation(transcript, [])

        assert verdict.success is False
        assert llm.prompts == []

    async def test_empty_llm_response_fails(self, transcript):
        verdict = await LLMJudge(ScriptedLLM("")).evaluate_conversation(
            transcript, [CONCISE]
        )

        assert verdict.success is False
        assert verdict.reason == "LLM returned an empty response."

    async def test_duplicate_criterion_names_raise(self, transcript):
        with pytest.raises(ValueError, match="unique"):
            await LLMJudge(ScriptedLLM("unused")).evaluate_conversation(
                transcript, [CONCISE, Criterion("concise", "again")]
            )

    async def test_prompt_contains_transcript_criteria_and_instructions(
        self, transcript
    ):
        llm = ScriptedLLM(_results())

        await LLMJudge(llm).evaluate_conversation(
            transcript, [SAY_DO_CONSISTENCY], instructions="Be brief."
        )

        prompt = llm.prompts[0]
        assert "Agent instructions:\nBe brief." in prompt
        assert "[user] What's the weather in Berlin?" in prompt
        assert '[tool call call_1] get_weather({"location": "Berlin"})' in prompt
        assert '[tool result call_1] get_weather -> {"temp_f": 70}' in prompt
        assert "[assistant] It is 70F in Berlin." in prompt
        assert f"- say_do_consistency: {SAY_DO_CONSISTENCY.description}" in prompt

    async def test_prompt_labels_tool_errors_and_omits_missing_ids(self):
        llm = ScriptedLLM(_results())
        events: list[RunEvent] = [
            FunctionCallEvent(name="send", arguments={}),
            FunctionCallOutputEvent(
                name="send", output={"error": "down"}, is_error=True
            ),
        ]

        await LLMJudge(llm).evaluate_conversation(events, [SAY_DO_CONSISTENCY])

        prompt = llm.prompts[0]
        assert "[tool call] send({})" in prompt
        assert '[tool error] send -> {"error": "down"}' in prompt
        assert "Agent instructions" not in prompt

    async def test_evaluate_single_message_against_intent(self):
        llm = ScriptedLLM(
            _results(
                {"name": "intent", "verdict": "pass", "score": 1.0, "reason": "Warm."}
            )
        )

        verdict = await LLMJudge(llm).evaluate(
            ChatMessageEvent(role="assistant", content="Hello there!"),
            intent="Friendly greeting",
        )

        assert verdict.success is True
        assert verdict.reason == "intent: Warm."
        assert "[assistant] Hello there!" in llm.prompts[0]
        assert "- intent: Friendly greeting" in llm.prompts[0]

    async def test_evaluate_empty_message_fails(self):
        llm = ScriptedLLM("unused")

        verdict = await LLMJudge(llm).evaluate(
            ChatMessageEvent(role="assistant", content=""), intent="Anything"
        )

        assert verdict.success is False
        assert verdict.reason == "The message is empty."
        assert llm.prompts == []

    async def test_evaluate_empty_intent_fails(self):
        llm = ScriptedLLM("unused")

        verdict = await LLMJudge(llm).evaluate(
            ChatMessageEvent(role="assistant", content="Hi"), intent=""
        )

        assert verdict.success is False
        assert llm.prompts == []

    @requires_gemini
    @pytest.mark.integration
    @pytest.mark.parametrize(
        ("criterion", "events", "instructions", "expected"),
        [
            (SAY_DO_CONSISTENCY, _SAY_DO_PASS, None, True),
            (SAY_DO_CONSISTENCY, _SAY_DO_FAIL, None, False),
            (STAYS_IN_SCOPE, _SCOPE_PASS, _WEATHER_INSTRUCTIONS, True),
            (STAYS_IN_SCOPE, _SCOPE_FAIL, _WEATHER_INSTRUCTIONS, False),
            (CONCISE, _CONCISE_PASS, None, True),
            (CONCISE, _CONCISE_FAIL, None, False),
            (RESPONDS_IN_USER_LANGUAGE, _LANGUAGE_PASS, None, True),
            (RESPONDS_IN_USER_LANGUAGE, _LANGUAGE_FAIL, None, False),
        ],
        ids=[
            "say_do-pass",
            "say_do-fail",
            "scope-pass",
            "scope-fail",
            "concise-pass",
            "concise-fail",
            "language-pass",
            "language-fail",
        ],
    )
    async def test_builtin_criteria_against_real_llm(
        self,
        criterion: Criterion,
        events: list[RunEvent],
        instructions: str | None,
        expected: bool,
    ):
        judge = LLMJudge(gemini.LLM(MODEL))

        verdict = await judge.evaluate_conversation(
            events, [criterion], instructions=instructions
        )

        assert verdict.success is expected, verdict.reason
        assert len(verdict.criteria) == 1
        assert verdict.criteria[0].name == criterion.name
