"""Unit tests for the multi-turn simulation: turn bounding, pass@k math,
variations and result aggregation. No real LLM is involved."""

import dataclasses
import json
from typing import AsyncIterator

import pytest
from getstream.video.rtc.track_util import PcmData

from vision_agents.core import Agent, User
from vision_agents.core.edge.types import Participant
from vision_agents.core.llm.llm import LLMResponseDelta, LLMResponseFinal
from vision_agents.testing import (
    ChatMessageEvent,
    JudgeError,
    LoopbackEdge,
    Scenario,
    SimulatedUser,
    SimulatedUserError,
    Simulation,
    generate_variations,
    pass_at_k,
    pass_pow_k,
)

from .fakes import (
    BookingLLM,
    CodecSTT,
    CodecTTS,
    ScriptedJudge,
    ScriptedLLM,
    user_done,
    user_says,
)


@pytest.fixture
def scenario() -> Scenario:
    return Scenario(
        name="reschedule",
        goal="Move appointment to Friday morning.",
        success=["appointment_rescheduled", "correct_time_confirmed"],
        context={"name": "Alice"},
        constraints=["Reject anything after 11am."],
    )


@pytest.fixture
def spoken_scenario(scenario) -> Scenario:
    return dataclasses.replace(scenario, mode="audio")


def spoken_agent(replies: list[str]) -> Agent:
    return Agent(
        edge=LoopbackEdge(),
        llm=BookingLLM(replies),
        stt=CodecSTT(),
        tts=CodecTTS(),
        agent_user=User(id="agent", name="Agent"),
        instructions="You book appointments.",
    )


def spoken_simulation(user_llm, **kwargs) -> Simulation:
    kwargs.setdefault("turn_timeout", 10.0)
    kwargs.setdefault("caller_tts", CodecTTS())
    kwargs.setdefault("caller_stt", CodecSTT())
    return Simulation(user_llm=user_llm, audio_settle=0.2, **kwargs)


class MuteTTS(CodecTTS):
    """Caller voice that fails to synthesise."""

    async def stream_audio(self, text: str, *args: object, **kwargs: object) -> PcmData:
        raise ConnectionError("voice offline")


class DeafSTT(CodecSTT):
    """Caller ears that fail on the first frame they are given."""

    async def process_audio(self, pcm_data: PcmData, participant: Participant) -> None:
        raise ConnectionError("ears offline")


class HangingUpUser(ScriptedLLM):
    """Simulated user whose second line is preceded by the agent leaving the call."""

    def __init__(self, replies: list[str], agent: Agent) -> None:
        super().__init__(replies)
        self._agent = agent
        self._calls = 0

    async def simple_response(
        self,
        text: str,
        participant: Participant | None = None,
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        self._calls += 1
        if self._calls == 2:
            await self._agent.close()
        async for item in super().simple_response(text, participant):
            yield item


class TestSimulatedUser:
    async def test_stops_when_goal_met(self, scenario):
        llm = ScriptedLLM([user_says("Hi"), user_says("Friday 10am?"), user_done()])
        user = SimulatedUser(llm, scenario, max_turns=10)

        assert await user.next_message(None) == "Hi"
        assert await user.next_message("Hello, how can I help?") == "Friday 10am?"
        assert await user.next_message("Booked.") is None
        assert await user.next_message("Anything else?") is None
        assert user.turns_taken == 2
        assert user.done is True

    async def test_stops_at_max_turns(self, scenario):
        llm = ScriptedLLM([user_says("still going")])
        user = SimulatedUser(llm, scenario, max_turns=3)

        messages = [await user.next_message(None)]
        for _ in range(5):
            messages.append(await user.next_message("ok"))

        assert messages == ["still going"] * 3 + [None] * 3
        assert user.turns_taken == 3
        assert user.done is False

    async def test_brief_is_in_instructions_and_reply_in_prompt(self, scenario):
        llm = ScriptedLLM([user_says("Hi"), user_done()])
        user = SimulatedUser(llm, scenario)
        await user.next_message(None)
        await user.next_message("Sure, when?")

        assert "Move appointment to Friday morning." in llm._instructions
        assert "- name: Alice" in llm._instructions
        assert "Sure, when?" in llm.history[2][1]

    async def test_empty_agent_reply_does_not_restart_conversation(self, scenario):
        llm = ScriptedLLM([user_says("Hi"), user_says("Hello?"), user_done()])
        user = SimulatedUser(llm, scenario)
        await user.next_message(None)
        await user.next_message("")

        assert "(the agent did not reply)" in llm.history[2][1]
        assert "opening message" not in llm.history[2][1]

    async def test_records_conversation_history(self, scenario):
        llm = ScriptedLLM([user_says("Hi"), user_says("Friday?"), user_done()])
        user = SimulatedUser(llm, scenario)
        await user.next_message(None)
        await user.next_message("When suits you?")

        roles = [role for role, _ in llm.history]
        assert roles == ["user", "assistant", "user", "assistant"]
        assert "When suits you?" in llm.history[2][1]
        assert llm.history[1][1] == user_says("Hi")

    async def test_llm_failure_raises(self, scenario):
        llm = ScriptedLLM(["x"], error=ConnectionError("boom"))
        user = SimulatedUser(llm, scenario)
        with pytest.raises(SimulatedUserError, match="LLM failed: boom"):
            await user.next_message(None)

    async def test_invalid_output_raises(self, scenario):
        user = SimulatedUser(ScriptedLLM(["not json"]), scenario)
        with pytest.raises(SimulatedUserError, match="invalid output"):
            await user.next_message(None)

    async def test_wrong_json_shape_raises(self, scenario):
        user = SimulatedUser(ScriptedLLM([json.dumps({"message": 1})]), scenario)
        with pytest.raises(SimulatedUserError, match="invalid output"):
            await user.next_message(None)

    async def test_timeout_raises(self, scenario):
        llm = ScriptedLLM([user_says("late")], delay=0.2)
        user = SimulatedUser(llm, scenario, turn_timeout=0.05)
        with pytest.raises(SimulatedUserError, match="did not respond"):
            await user.next_message(None)

    def test_rejects_bad_bounds(self, scenario):
        with pytest.raises(ValueError, match="max_turns"):
            SimulatedUser(ScriptedLLM(["x"]), scenario, max_turns=0)
        with pytest.raises(ValueError, match="turn_timeout"):
            SimulatedUser(ScriptedLLM(["x"]), scenario, turn_timeout=0)


class TestPassAtK:
    def test_all_pass(self):
        assert pass_at_k(3, 3, 3) == 1.0
        assert pass_pow_k(3, 3, 3) == 1.0

    def test_none_pass(self):
        assert pass_at_k(5, 0, 1) == 0.0
        assert pass_pow_k(5, 0, 1) == 0.0

    def test_k_one_is_pass_rate(self):
        assert pass_at_k(5, 2, 1) == pytest.approx(0.4)
        assert pass_pow_k(5, 2, 1) == pytest.approx(0.4)

    def test_unbiased_estimates(self):
        assert pass_at_k(4, 2, 2) == pytest.approx(5 / 6)
        assert pass_pow_k(4, 2, 2) == pytest.approx(1 / 6)

    def test_any_pass_with_n_equal_k(self):
        assert pass_at_k(3, 1, 3) == 1.0
        assert pass_pow_k(3, 1, 3) == 0.0

    @pytest.mark.parametrize("n, c, k", [(2, 0, 3), (3, 4, 1), (3, 1, 0), (3, -1, 1)])
    def test_invalid_inputs_raise(self, n, c, k):
        with pytest.raises(ValueError):
            pass_at_k(n, c, k)
        with pytest.raises(ValueError):
            pass_pow_k(n, c, k)


class TestGenerateVariations:
    async def test_original_first_then_rewordings(self, scenario):
        llm = ScriptedLLM(
            [
                json.dumps(
                    {
                        "variations": [
                            {
                                "goal": "Get my appointment moved to Friday before noon.",
                                "constraints": ["Turn down any slot later than 11am."],
                            },
                            {
                                "goal": "Reschedule to a Friday morning slot.",
                                "constraints": ["Nothing after 11am works for me."],
                            },
                        ]
                    }
                )
            ]
        )
        variants = await generate_variations(llm, scenario, 3)

        assert len(variants) == 3
        assert variants[0] == scenario
        assert variants[1].goal == "Get my appointment moved to Friday before noon."
        assert variants[1].constraints == ["Turn down any slot later than 11am."]
        assert variants[2].goal == "Reschedule to a Friday morning slot."
        assert all(v.context == scenario.context for v in variants)
        assert all(v.success == scenario.success for v in variants)

    async def test_single_variation_skips_llm(self, scenario):
        llm = ScriptedLLM(["x"], error=RuntimeError("must not be called"))
        assert await generate_variations(llm, scenario, 1) == [scenario]

    async def test_duplicate_variation_rejected(self, scenario):
        llm = ScriptedLLM(
            [
                json.dumps(
                    {
                        "variations": [
                            {"goal": "Same", "constraints": ["Same rule."]},
                            {"goal": "Same", "constraints": ["Same rule."]},
                        ]
                    }
                )
            ]
        )
        with pytest.raises(ValueError, match="duplicates"):
            await generate_variations(llm, scenario, 3)

    async def test_variation_equal_to_original_rejected(self, scenario):
        llm = ScriptedLLM(
            [
                json.dumps(
                    {
                        "variations": [
                            {"goal": scenario.goal, "constraints": scenario.constraints}
                        ]
                    }
                )
            ]
        )
        with pytest.raises(ValueError, match="duplicates"):
            await generate_variations(llm, scenario, 2)

    async def test_too_few_variations_raises(self, scenario):
        llm = ScriptedLLM(
            [json.dumps({"variations": [{"goal": "only one", "constraints": ["x"]}]})]
        )
        with pytest.raises(ValueError, match="Expected 2 variation"):
            await generate_variations(llm, scenario, 3)

    async def test_constraint_count_must_match(self, scenario):
        llm = ScriptedLLM(
            [json.dumps({"variations": [{"goal": "g", "constraints": []}]})]
        )
        with pytest.raises(ValueError, match="keep 1 constraint"):
            await generate_variations(llm, scenario, 2)


class TestSimulation:
    async def test_single_conversation_result(self, scenario):
        user_llm = ScriptedLLM(
            [
                user_says("Hi, I need to move my appointment."),
                user_says("Friday 10am?"),
                user_done(),
            ]
        )
        agent_llm = BookingLLM(["What time suits you?", "Booked Friday 10am."])
        judge = ScriptedJudge([True, True])

        result = await Simulation(user_llm=user_llm).run(
            agent_llm, scenario, judge, instructions="You book appointments."
        )

        assert result.passed is True
        assert result.pass_rate == 1.0
        assert result.pass_at_k == 1.0
        assert result.pass_pow_k == 1.0
        assert len(result.trials) == 1
        trial = result.trials[0]
        assert trial.turn_count == 2
        assert trial.passed is True
        assert trial.valid is True
        assert [t.user_message for t in trial.turns] == [
            "Hi, I need to move my appointment.",
            "Friday 10am?",
        ]
        assert [t.agent_reply for t in trial.turns] == [
            "What time suits you?",
            "Booked Friday 10am.",
        ]
        assert len(trial.latencies_ms) == 2
        assert all(latency >= 0 for latency in trial.latencies_ms)
        assert [c.name for c in trial.tool_calls] == ["book_slot", "book_slot"]
        assert trial.tool_calls[0].arguments == {"day": "Friday", "time": "10am"}
        assert set(trial.verdicts) == {
            "appointment_rescheduled",
            "correct_time_confirmed",
        }
        roles = [e.role for e in trial.transcript if isinstance(e, ChatMessageEvent)]
        assert roles == ["user", "assistant", "user", "assistant"]
        assert agent_llm._instructions == "You book appointments."

    async def test_judge_sees_transcript_and_criterion(self, scenario):
        user_llm = ScriptedLLM([user_says("Move me to Friday"), user_done()])
        agent_llm = BookingLLM(["Done."])

        def transcript_complete(event: ChatMessageEvent, intent: str) -> bool:
            return (
                "[user] Move me to Friday" in event.content
                and "[agent called tool book_slot]" in event.content
                and "[assistant] Done." in event.content
                and scenario.goal in intent
            )

        def names_criterion(event: ChatMessageEvent, intent: str) -> bool:
            return "correct_time_confirmed" in intent

        judge = ScriptedJudge([transcript_complete, names_criterion])
        result = await Simulation(user_llm=user_llm).run(agent_llm, scenario, judge)

        assert result.passed is True, result.summary()

    async def test_tool_only_turn_continues_conversation(self, scenario):
        user_llm = ScriptedLLM(
            [user_says("Book Friday"), user_says("Did it work?"), user_done()]
        )
        agent_llm = BookingLLM(["", "Yes, booked Friday 10am."])

        result = await Simulation(user_llm=user_llm).run(
            agent_llm, scenario, ScriptedJudge()
        )

        trial = result.trials[0]
        assert trial.turn_count == 2
        assert [t.agent_reply for t in trial.turns] == [
            None,
            "Yes, booked Friday 10am.",
        ]
        assert "(the agent did not reply)" in user_llm.history[2][1]
        assert result.passed is True

    async def test_failed_criterion_fails_trial(self, scenario):
        user_llm = ScriptedLLM([user_says("Hi"), user_done()])
        result = await Simulation(user_llm=user_llm).run(
            ScriptedLLM(["No."]), scenario, ScriptedJudge([True, False])
        )
        trial = result.trials[0]
        assert trial.valid is True
        assert trial.passed is False
        assert trial.verdicts["correct_time_confirmed"].success is False
        assert result.passed is False
        assert result.pass_rate == 0.0

    async def test_repeat_reports_pass_at_k_and_pass_pow_k(self, scenario):
        repeated = Scenario(name="r", goal=scenario.goal, success=["ok"], repeat=3)
        judge = ScriptedJudge([True, False, True])

        result = await Simulation(
            user_llm=lambda: ScriptedLLM([user_says("Hi"), user_done()])
        ).run(lambda: ScriptedLLM(["Reply"]), repeated, judge)

        assert len(result.trials) == 3
        assert [t.repeat for t in result.trials] == [0, 1, 2]
        assert [t.passed for t in result.trials] == [True, False, True]
        assert result.k == 3
        assert result.pass_rate == pytest.approx(2 / 3)
        assert result.pass_at_k == 1.0
        assert result.pass_pow_k == 0.0
        assert result.passed is False

    async def test_judge_error_marks_trial_invalid_not_failed(self, scenario):
        repeated = Scenario(name="r", goal=scenario.goal, success=["ok"], repeat=2)
        judge = ScriptedJudge([JudgeError("quota exceeded"), True])

        result = await Simulation(
            user_llm=lambda: ScriptedLLM([user_says("Hi"), user_done()])
        ).run(lambda: ScriptedLLM(["Reply"]), repeated, judge)

        invalid, valid = result.trials
        assert invalid.valid is False
        assert invalid.passed is False
        assert "quota exceeded" in invalid.error
        assert valid.passed is True
        assert result.invalid_trials == [invalid]
        assert result.passed is True
        assert result.pass_rate == 1.0
        assert result.pass_at_k is None
        assert result.pass_pow_k is None

    async def test_unexpected_judge_exception_marks_trial_invalid(self, scenario):
        result = await Simulation(
            user_llm=ScriptedLLM([user_says("Hi"), user_done()])
        ).run(ScriptedLLM(["Reply"]), scenario, ScriptedJudge([RuntimeError("down")]))
        trial = result.trials[0]
        assert trial.valid is False
        assert "down" in trial.error
        assert result.pass_rate is None

    async def test_simulated_user_failure_marks_trial_invalid(self, scenario):
        result = await Simulation(user_llm=ScriptedLLM(["garbage"])).run(
            ScriptedLLM(["Reply"]), scenario, ScriptedJudge()
        )
        trial = result.trials[0]
        assert trial.valid is False
        assert trial.turn_count == 0
        assert result.passed is False
        assert result.pass_rate is None

    async def test_agent_timeout_fails_trial(self, scenario):
        user_llm = ScriptedLLM([user_says("Hi"), user_done()])
        agent_llm = ScriptedLLM(["slow"], delay=0.2)
        judge = ScriptedJudge([RuntimeError("judge must not run")])

        result = await Simulation(user_llm=user_llm, turn_timeout=0.05).run(
            agent_llm, scenario, judge
        )

        trial = result.trials[0]
        assert trial.valid is True
        assert trial.passed is False
        assert "did not reply" in trial.error
        assert trial.verdicts == {}

    async def test_max_turns_bounds_conversation(self, scenario):
        user_llm = ScriptedLLM([user_says("again")])
        result = await Simulation(user_llm=user_llm, max_turns=4).run(
            ScriptedLLM(["Reply"]), scenario, ScriptedJudge()
        )
        assert result.trials[0].turn_count == 4

    async def test_variations_run_original_first(self, scenario):
        varied = Scenario(
            name="v",
            goal=scenario.goal,
            success=["ok"],
            constraints=scenario.constraints,
            variations=2,
        )
        variations_json = json.dumps(
            {
                "variations": [
                    {"goal": "Reworded goal.", "constraints": ["Reworded rule."]}
                ]
            }
        )
        user_llms = iter(
            [
                ScriptedLLM([variations_json]),
                ScriptedLLM([user_says("Hi"), user_done()]),
                ScriptedLLM([user_says("Hello"), user_done()]),
            ]
        )

        result = await Simulation(user_llm=lambda: next(user_llms)).run(
            lambda: ScriptedLLM(["Reply"]), varied, ScriptedJudge()
        )

        assert len(result.trials) == 2
        assert [t.variation for t in result.trials] == [0, 1]
        assert result.trials[0].scenario.goal == scenario.goal
        assert result.trials[1].scenario.goal == "Reworded goal."
        assert result.trials[1].scenario.constraints == ["Reworded rule."]
        assert result.passed is True

    async def test_pass_metrics_are_averaged_per_variation(self, scenario):
        varied = Scenario(
            name="v", goal=scenario.goal, success=["ok"], variations=2, repeat=2
        )
        variations_json = json.dumps(
            {"variations": [{"goal": "Reworded goal.", "constraints": []}]}
        )
        user_llms = iter(
            [ScriptedLLM([variations_json])]
            + [ScriptedLLM([user_says("Hi"), user_done()]) for _ in range(4)]
        )
        judge = ScriptedJudge([True, True, True, False])

        result = await Simulation(user_llm=lambda: next(user_llms)).run(
            lambda: ScriptedLLM(["Reply"]), varied, judge
        )

        assert [(t.variation, t.passed) for t in result.trials] == [
            (0, True),
            (0, True),
            (1, True),
            (1, False),
        ]
        assert result.pass_rate == pytest.approx(0.75)
        assert result.pass_at_k == pytest.approx(1.0)
        assert result.pass_pow_k == pytest.approx(0.5)

    async def test_pass_metrics_none_when_a_variation_lacks_valid_trials(
        self, scenario
    ):
        varied = Scenario(
            name="v", goal=scenario.goal, success=["ok"], variations=2, repeat=2
        )
        variations_json = json.dumps(
            {"variations": [{"goal": "Reworded goal.", "constraints": []}]}
        )
        user_llms = iter(
            [ScriptedLLM([variations_json])]
            + [ScriptedLLM([user_says("Hi"), user_done()]) for _ in range(4)]
        )
        judge = ScriptedJudge([True, True, JudgeError("down"), True])

        result = await Simulation(user_llm=lambda: next(user_llms)).run(
            lambda: ScriptedLLM(["Reply"]), varied, judge
        )

        assert len(result.invalid_trials) == 1
        assert result.pass_rate == pytest.approx(1.0)
        assert result.pass_at_k is None
        assert result.pass_pow_k is None

    async def test_instance_rejected_for_multiple_conversations(self, scenario):
        repeated = Scenario(name="r", goal=scenario.goal, success=["ok"], repeat=2)
        simulation = Simulation(user_llm=lambda: ScriptedLLM([user_done()]))
        with pytest.raises(ValueError, match="agent_or_llm must be a factory"):
            await simulation.run(ScriptedLLM(["Reply"]), repeated, ScriptedJudge())

        simulation = Simulation(user_llm=ScriptedLLM([user_done()]))
        with pytest.raises(ValueError, match="user_llm must be a factory"):
            await simulation.run(
                lambda: ScriptedLLM(["Reply"]), repeated, ScriptedJudge()
            )

    async def test_summary_lists_trials_and_verdicts(self, scenario):
        user_llm = ScriptedLLM([user_says("Hi"), user_done()])
        result = await Simulation(user_llm=user_llm).run(
            ScriptedLLM(["Reply"]), scenario, ScriptedJudge([True, False])
        )
        summary = result.summary()
        assert "Scenario 'reschedule': FAIL" in summary
        assert "0/1 valid trials passed" in summary
        assert "correct_time_confirmed: fail" in summary
        assert "[user] Hi" in summary


class TestSpokenSimulation:
    async def test_judge_reads_what_the_caller_heard(self, spoken_scenario):
        user_llm = ScriptedLLM([user_says("Move me to Friday"), user_done()])

        def heard_transcript(event: ChatMessageEvent, intent: str) -> bool:
            return (
                "[user] Move me to Friday" in event.content
                and "[agent called tool book_slot]" in event.content
                and "[assistant] Booked Friday 10am." in event.content
            )

        judge = ScriptedJudge([heard_transcript, heard_transcript])
        result = await spoken_simulation(user_llm).run(
            spoken_agent(["Booked Friday 10am."]), spoken_scenario, judge
        )

        assert result.passed, result.summary()
        trial = result.trials[0]
        assert trial.turn_count == 1
        turn = trial.turns[0]
        assert turn.user_message == "Move me to Friday"
        assert turn.agent_reply == "Booked Friday 10am."
        assert turn.intended_reply == "Booked Friday 10am."
        assert turn.voice_to_voice_ms is not None
        assert 0 < turn.voice_to_voice_ms < 5000
        assert turn.latency_ms >= turn.voice_to_voice_ms
        assert trial.voice_to_voice_ms == [turn.voice_to_voice_ms]
        assert [c.name for c in trial.tool_calls] == ["book_slot"]
        assert "Booked Friday 10am." in user_llm.history[2][1]

    async def test_summary_shows_intended_text_and_latency(self, spoken_scenario):
        user_llm = ScriptedLLM([user_says("Hi"), user_done()])
        result = await spoken_simulation(user_llm).run(
            spoken_agent(["Done."]), spoken_scenario, ScriptedJudge()
        )
        summary = result.summary()
        assert "[assistant] Done." in summary
        assert "intended (what the agent meant to say):" in summary
        assert "voice-to-voice ms:" in summary

    async def test_silent_agent_fails_trial(self, spoken_scenario):
        user_llm = ScriptedLLM([user_says("Hi"), user_done()])
        judge = ScriptedJudge([RuntimeError("judge must not be called")])

        result = await spoken_simulation(user_llm, turn_timeout=0.5).run(
            spoken_agent([""]), spoken_scenario, judge
        )

        trial = result.trials[0]
        assert trial.valid is True
        assert trial.passed is False
        assert "did not reply within 0.5s" in trial.error
        assert trial.verdicts == {}

    async def test_caller_voice_failure_marks_trial_invalid(self, spoken_scenario):
        user_llm = ScriptedLLM([user_says("Hi"), user_done()])
        judge = ScriptedJudge([RuntimeError("judge must not be called")])

        result = await spoken_simulation(user_llm, caller_tts=MuteTTS()).run(
            spoken_agent(["Done."]), spoken_scenario, judge
        )

        trial = result.trials[0]
        assert trial.valid is False
        assert trial.turn_count == 0
        assert "voice failed: voice offline" in trial.error
        assert result.pass_rate is None

    async def test_caller_ears_failure_marks_trial_invalid(self, spoken_scenario):
        user_llm = ScriptedLLM([user_says("Hi"), user_done()])
        judge = ScriptedJudge([RuntimeError("judge must not be called")])

        result = await spoken_simulation(user_llm, caller_stt=DeafSTT()).run(
            spoken_agent(["Done."]), spoken_scenario, judge
        )

        trial = result.trials[0]
        assert trial.valid is False
        assert "ears failed: ears offline" in trial.error
        assert result.pass_rate is None

    async def test_agent_leaving_between_turns_fails_trial(self, spoken_scenario):
        agent = spoken_agent(["Bye."])
        user_llm = HangingUpUser(
            [user_says("Hi"), user_says("Still there?"), user_done()], agent
        )

        result = await spoken_simulation(user_llm).run(
            agent, spoken_scenario, ScriptedJudge()
        )

        trial = result.trials[0]
        assert trial.valid is True
        assert trial.passed is False
        assert trial.turn_count == 1
        assert trial.turns[0].agent_reply == "Bye."
        assert "call ended" in trial.error

    async def test_fails_fast_without_caller_voice(self, spoken_scenario):
        built: list[Agent] = []

        def build_agent() -> Agent:
            built.append(spoken_agent(["Done."]))
            return built[-1]

        simulation = Simulation(
            user_llm=ScriptedLLM([user_done()]), caller_stt=CodecSTT()
        )
        with pytest.raises(ValueError, match="no caller TTS.*caller_tts"):
            await simulation.run(build_agent, spoken_scenario, ScriptedJudge())

        simulation = Simulation(
            user_llm=ScriptedLLM([user_done()]), caller_tts=CodecTTS()
        )
        with pytest.raises(ValueError, match="no caller STT.*caller_stt"):
            await simulation.run(build_agent, spoken_scenario, ScriptedJudge())
        assert built == []

    async def test_fails_fast_when_scenario_names_unknown_plugin(self, spoken_scenario):
        scenario = dataclasses.replace(
            spoken_scenario, caller_tts="no_such_plugin", caller_stt="codec"
        )
        simulation = Simulation(user_llm=ScriptedLLM([user_done()]))
        with pytest.raises(ValueError, match="'no_such_plugin' is not installed"):
            await simulation.run(
                lambda: spoken_agent(["Done."]), scenario, ScriptedJudge()
            )

    async def test_requires_agent_with_loopback_edge(self, spoken_scenario):
        simulation = spoken_simulation(ScriptedLLM([user_says("Hi"), user_done()]))
        with pytest.raises(ValueError, match="LoopbackEdge"):
            await simulation.run(
                ScriptedLLM(["Reply"]), spoken_scenario, ScriptedJudge()
            )

    async def test_caller_instances_rejected_for_multiple_conversations(
        self, spoken_scenario
    ):
        repeated = dataclasses.replace(spoken_scenario, repeat=2)
        simulation = spoken_simulation(lambda: ScriptedLLM([user_done()]))
        with pytest.raises(ValueError, match="caller_tts must be a factory"):
            await simulation.run(
                lambda: spoken_agent(["Done."]), repeated, ScriptedJudge()
            )


class TestSimulateFixture:
    async def test_runs_scenario_relative_to_test_file(self, simulate):
        result = await simulate(
            "../test_assets/scenarios/reschedule-appointment.yaml",
            instructions="You book appointments.",
        )
        assert result.scenario.name == "reschedule-appointment"
        assert result.passed is True
        assert result.trials[0].turn_count == 1
        assert set(result.trials[0].verdicts) == {
            "appointment_rescheduled",
            "correct_time_confirmed",
            "identity_verified",
        }

    async def test_accepts_scenario_object(self, simulate, scenario):
        result = await simulate(scenario)
        assert result.scenario is scenario
        assert result.passed is True
