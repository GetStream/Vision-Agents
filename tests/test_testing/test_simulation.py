"""Unit tests for scenario loading, the Simulator and verdict parsing.

No real LLM is involved: ``ScriptedLLM`` plays every role from the prompt it
receives, so conversations, verdicts and reports are deterministic.
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Optional

import pytest
from getstream.video.rtc import AudioStreamTrack
from vision_agents.core import Agent, User
from vision_agents.core.edge.edge_transport import EdgeTransport
from vision_agents.core.llm.llm import LLM, LLMResponseDelta, LLMResponseFinal
from vision_agents.core.tts import TTS
from vision_agents.testing import (
    Scenario,
    ScenarioError,
    SimulationReport,
    Simulator,
    find_scenarios,
    load_scenario,
    parse_verdict,
)

AGENT_INSTRUCTIONS = "You are the stub weather agent."
BRIEF = "You want to know today's weather in Amsterdam."
JUDGE = "stub/judge"


class StubEdge(EdgeTransport):
    async def authenticate(self, user):
        pass

    async def create_call(self, call_id, **kwargs):
        raise NotImplementedError

    def create_audio_track(self):
        return AudioStreamTrack(
            audio_buffer_size_ms=300_000, sample_rate=48000, channels=2
        )

    def open_demo(self, *args, **kwargs):
        pass

    async def join(self, agent, call, **kwargs):
        raise NotImplementedError

    async def publish_tracks(self, audio_track, video_track):
        raise NotImplementedError

    async def close(self):
        pass

    async def create_conversation(self, call, user, instructions):
        raise NotImplementedError

    def add_track_subscriber(self, track_id):
        return None

    async def send_custom_event(self, data):
        pass


class StubTTS(TTS):
    async def stream_audio(self, *args, **kwargs):
        return b""

    async def stop_audio(self):
        pass


class ScriptedLLM(LLM):
    """Answers as agent, caller, judge or rewriter depending on the prompt.

    Args:
        caller_lines: What the caller says on successive turns; ``[END]`` ends it.
        judge_reply: Raw judge output; defaults to pass unless the criterion
            mentions "impossible".
        error: Raise this from every call instead of answering.
    """

    def __init__(
        self,
        caller_lines: Optional[list[str]] = None,
        judge_reply: Optional[str] = None,
        error: Optional[str] = None,
        swallow_error: bool = False,
    ):
        super().__init__()
        self._caller_lines = caller_lines or ["What's the weather?", "Thanks [END]"]
        self._judge_reply = judge_reply
        self._error = error
        self._swallow_error = swallow_error
        self._caller_turns = 0

    async def simple_response(
        self, text: str, participant=None
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        if self._error and self._swallow_error:
            self.on_llm_error(error=RuntimeError(self._error))
            yield LLMResponseFinal(text="")
            return
        if self._error:
            raise RuntimeError(self._error)
        yield LLMResponseFinal(text=await self._reply(text))

    async def _reply(self, text: str) -> str:
        if self._instructions == AGENT_INSTRUCTIONS:
            if "get_weather" in self.function_registry.functions:
                await self.function_registry.call_function(
                    "get_weather", {"city": "Amsterdam"}
                )
            return "It is sunny in Amsterdam today."
        if "Criterion:" in text:
            if self._judge_reply is not None:
                return self._judge_reply
            verdict = "fail" if "impossible" in text else "pass"
            return json.dumps({"verdict": verdict, "reason": "stub judge"})
        if "Brief:" in text:
            return json.dumps(["Rewrite 1", "Rewrite 2"])
        line = self._caller_lines[min(self._caller_turns, len(self._caller_lines) - 1)]
        self._caller_turns += 1
        return line


def _agent_factory(llm_factory):
    async def create_agent() -> Agent:
        return Agent(
            edge=StubEdge(),
            agent_user=User(name="stub", id="stub"),
            instructions=AGENT_INSTRUCTIONS,
            llm=llm_factory(),
            tts=StubTTS(),
        )

    return create_agent


def _scenario(
    name: str = "weather",
    criteria: Optional[list[str]] = None,
    max_turns: int = 12,
    variations: int = 1,
) -> Scenario:
    return Scenario(
        name=name,
        scenario=BRIEF,
        criteria=criteria or ["The agent tells the user the weather"],
        max_turns=max_turns,
        variations=variations,
    )


class TestLoadScenario:
    def test_loads_fields_and_defaults(self, tmp_path: Path):
        path = tmp_path / "refund.toml"
        path.write_text('scenario = "You want a refund."\ncriteria = ["Polite"]\n')
        scenario = load_scenario(path)
        assert scenario.name == "refund"
        assert scenario.scenario == "You want a refund."
        assert scenario.criteria == ["Polite"]
        assert scenario.max_turns == 12
        assert scenario.variations == 1
        assert scenario.path == path

    def test_explicit_fields_override_defaults(self, tmp_path: Path):
        path = tmp_path / "x.toml"
        path.write_text(
            'name = "Refund flow"\nscenario = "s"\ncriteria = ["c"]\n'
            "max_turns = 3\nvariations = 2\n"
        )
        scenario = load_scenario(path)
        assert scenario.name == "Refund flow"
        assert scenario.max_turns == 3
        assert scenario.variations == 2

    @pytest.mark.parametrize(
        ("content", "message"),
        [
            ('criteria = ["c"]\n', "scenario is required"),
            ('scenario = "s"\n', "criteria is required"),
            ('scenario = "s"\ncriteria = "c"\n', "criteria is required"),
            ('scenario = "s"\ncriteria = []\n', "non-empty list"),
            ('scenario = "s"\ncriteria = ["c"]\nmax_turns = 0\n', "max_turns"),
            ('scenario = "s"\ncriteria = ["c"]\nvariations = "2"\n', "variations"),
            ('scenario = "s"\ncriteria = ["c"\n', "failed to parse"),
        ],
    )
    def test_rejects_malformed_files(self, tmp_path: Path, content: str, message: str):
        path = tmp_path / "bad.toml"
        path.write_text(content)
        with pytest.raises(ScenarioError, match=message) as exc_info:
            load_scenario(path)
        assert "bad.toml" in str(exc_info.value)

    def test_find_scenarios_returns_file_or_sorted_directory(self, tmp_path: Path):
        (tmp_path / "b.toml").write_text("")
        (tmp_path / "a.toml").write_text("")
        (tmp_path / "notes.md").write_text("")
        nested = tmp_path / "nested"
        nested.mkdir()
        (nested / "c.toml").write_text("")
        assert find_scenarios(tmp_path) == [tmp_path / "a.toml", tmp_path / "b.toml"]
        assert find_scenarios(tmp_path / "b.toml") == [tmp_path / "b.toml"]

    def test_find_scenarios_missing_path_raises(self, tmp_path: Path):
        with pytest.raises(ScenarioError, match="does not exist"):
            find_scenarios(tmp_path / "nowhere")


class TestSimulator:
    @pytest.fixture
    def simulator(self) -> Simulator:
        return Simulator(_agent_factory(ScriptedLLM), ScriptedLLM, judge_target=JUDGE)

    async def test_passing_scenario(self, simulator: Simulator):
        report = await simulator.run([_scenario()])

        assert report.state == "passed"
        assert report.exit_code == 0
        assert report.judge_target == JUDGE
        run = report.runs[0]
        assert run.state == "passed"
        assert (run.cases, run.passed, run.failed, run.errored) == (1, 1, 0, 0)
        case = run.conversations[0]
        assert case.turns == 2
        assert case.ended == "complete"
        assert case.passed is True
        assert case.verdict == "All criteria met."
        assert [(line.caller, line.text) for line in case.transcript] == [
            (True, "What's the weather?"),
            (False, "It is sunny in Amsterdam today."),
            (True, "Thanks"),
            (False, "It is sunny in Amsterdam today."),
        ]
        assert all(
            line.latency_ms is not None for line in case.transcript if not line.caller
        )
        assert run.p50_latency_ms is not None
        assert run.mean_turns == 2

    async def test_failed_criterion_fails_the_case(self, simulator: Simulator):
        criteria = ["The agent tells the weather", "The agent does the impossible"]
        report = await simulator.run([_scenario(criteria=criteria)])

        assert report.state == "failed"
        assert report.exit_code == 1
        run = report.runs[0]
        assert run.failed_criteria == ["The agent does the impossible"]
        case = run.conversations[0]
        assert case.passed is False
        assert case.verdict == "The agent does the impossible: stub judge"
        assert [v.passed for v in case.criteria] == [True, False]

    async def test_caller_that_never_ends_stops_at_max_turns(self):
        def caller():
            return ScriptedLLM(caller_lines=["Tell me more."])

        simulator = Simulator(_agent_factory(ScriptedLLM), caller, judge_target=JUDGE)
        report = await simulator.run([_scenario(max_turns=3)])

        case = report.runs[0].conversations[0]
        assert case.turns == 3
        assert case.ended == "turns"
        assert len(case.transcript) == 6

    @pytest.mark.parametrize(
        ("caller_lines", "message"),
        [
            ([""], "empty message"),
            (["[END]"], "before saying anything"),
        ],
    )
    async def test_silent_caller_marks_case_errored(
        self, caller_lines: list[str], message: str
    ):
        def caller():
            return ScriptedLLM(caller_lines=caller_lines)

        simulator = Simulator(_agent_factory(ScriptedLLM), caller, judge_target=JUDGE)
        report = await simulator.run([_scenario()])

        case = report.runs[0].conversations[0]
        assert case.state == "errored"
        assert case.error is not None
        assert message in case.error
        assert case.transcript == []
        assert report.exit_code == 2

    async def test_caller_ending_with_bare_end_token_completes(self):
        def caller():
            return ScriptedLLM(caller_lines=["Hello?", "[END]"])

        simulator = Simulator(_agent_factory(ScriptedLLM), caller, judge_target=JUDGE)
        report = await simulator.run([_scenario()])

        case = report.runs[0].conversations[0]
        assert case.state == "passed"
        assert case.ended == "complete"
        assert case.turns == 1
        assert len(case.transcript) == 2

    async def test_repeat_and_variations(self):
        simulator = Simulator(
            _agent_factory(ScriptedLLM),
            ScriptedLLM,
            judge_target=JUDGE,
            repeat=2,
            variations=3,
        )
        report = await simulator.run([_scenario(variations=1)])

        run = report.runs[0]
        assert run.variations == 3
        assert run.cases == 6
        assert report.repeat == 2
        assert [(c.variation, c.attempt, c.scenario) for c in run.conversations] == [
            (0, 0, BRIEF),
            (0, 1, BRIEF),
            (1, 0, "Rewrite 1"),
            (1, 1, "Rewrite 1"),
            (2, 0, "Rewrite 2"),
            (2, 1, "Rewrite 2"),
        ]

    async def test_scenario_variations_used_when_not_overridden(self, simulator):
        report = await simulator.run([_scenario(variations=2)])
        assert [c.scenario for c in report.runs[0].conversations] == [
            BRIEF,
            "Rewrite 1",
        ]

    async def test_too_few_rewrites_errors_the_run(self):
        simulator = Simulator(
            _agent_factory(ScriptedLLM), ScriptedLLM, judge_target=JUDGE, variations=5
        )
        report = await simulator.run([_scenario()])

        run = report.runs[0]
        assert run.state == "errored"
        assert run.cases == 0
        assert run.error is not None
        assert "asked for 4 variations" in run.error
        assert report.exit_code == 2

    @pytest.mark.parametrize("swallow_error", [False, True])
    async def test_agent_error_marks_case_errored(self, swallow_error: bool):
        def broken_agent_llm():
            return ScriptedLLM(error="provider exploded", swallow_error=swallow_error)

        simulator = Simulator(
            _agent_factory(broken_agent_llm), ScriptedLLM, judge_target=JUDGE
        )
        report = await simulator.run([_scenario()])

        case = report.runs[0].conversations[0]
        assert case.state == "errored"
        assert case.passed is None
        assert case.verdict is None
        assert case.error is not None
        assert "provider exploded" in case.error
        assert case.criteria == []
        assert report.state == "errored"
        assert report.exit_code == 2

    async def test_unparsable_judge_output_marks_case_errored(self):
        def judge():
            return ScriptedLLM(judge_reply="I think it went well")

        simulator = Simulator(_agent_factory(ScriptedLLM), judge, judge_target=JUDGE)
        report = await simulator.run([_scenario()])

        case = report.runs[0].conversations[0]
        assert case.state == "errored"
        assert case.ended == "complete"
        assert case.error is not None
        assert "Could not parse JSON" in case.error
        assert len(case.transcript) == 4

    async def test_tool_calls_are_recorded_on_agent_lines(self):
        def agent_llm():
            llm = ScriptedLLM()

            @llm.register_function(description="Weather lookup")
            async def get_weather(city: str) -> dict:
                return {"city": city, "condition": "sunny"}

            return llm

        simulator = Simulator(
            _agent_factory(agent_llm), ScriptedLLM, judge_target=JUDGE
        )
        report = await simulator.run([_scenario()])

        agent_line = report.runs[0].conversations[0].transcript[1]
        assert [call.to_dict() for call in agent_line.tool_calls] == [
            {
                "name": "get_weather",
                "arguments": {"city": "Amsterdam"},
                "output": {"city": "Amsterdam", "condition": "sunny"},
                "is_error": False,
            }
        ]

    async def test_on_case_is_called_per_conversation(self):
        seen = []
        simulator = Simulator(
            _agent_factory(ScriptedLLM), ScriptedLLM, judge_target=JUDGE, repeat=2
        )
        await simulator.run(
            [_scenario(), _scenario(name="second")],
            on_case=lambda scenario, case: seen.append((scenario.name, case.attempt)),
        )
        assert seen == [("weather", 0), ("weather", 1), ("second", 0), ("second", 1)]

    async def test_report_dict_mirrors_simulation_run_schema(self, simulator):
        report = await simulator.run([_scenario()])
        data = report.to_dict()

        assert set(data) == {
            "state",
            "mode",
            "judge_target",
            "repeat",
            "cases",
            "passed",
            "failed",
            "errored",
            "started_at",
            "finished_at",
            "runs",
        }
        run = data["runs"][0]
        assert run["mode"] == "text"
        assert run["judge_target"] == JUDGE
        assert {
            "id",
            "name",
            "scenario",
            "criteria",
            "state",
            "cases",
            "passed",
            "failed",
            "started_at",
            "finished_at",
            "conversations",
        } <= set(run)
        case = run["conversations"][0]
        assert {
            "id",
            "variation",
            "scenario",
            "state",
            "transcript",
            "turns",
            "passed",
            "verdict",
            "ended",
            "started_at",
            "finished_at",
        } <= set(case)
        assert case["transcript"][0] == {
            "caller": True,
            "text": "What's the weather?",
            "at": case["transcript"][0]["at"],
            "latency_ms": None,
            "tool_calls": [],
        }
        json.dumps(data)

    @pytest.mark.parametrize("kwargs", [{"repeat": 0}, {"variations": 0}])
    def test_rejects_invalid_counts(self, kwargs):
        with pytest.raises(ValueError):
            Simulator(
                _agent_factory(ScriptedLLM), ScriptedLLM, judge_target=JUDGE, **kwargs
            )


class TestSimulationReport:
    async def test_errored_takes_precedence_over_failed(self):
        simulator = Simulator(
            _agent_factory(ScriptedLLM), ScriptedLLM, judge_target=JUDGE
        )
        report = await simulator.run(
            [
                _scenario(name="failing", criteria=["The agent does the impossible"]),
                _scenario(name="erroring", variations=5),
            ]
        )

        assert [run.state for run in report.runs] == ["failed", "errored"]
        assert report.state == "errored"
        assert report.exit_code == 2
        assert (report.cases, report.passed, report.failed, report.errored) == (
            1,
            0,
            1,
            0,
        )

    def test_empty_report_passes(self):
        now = datetime.now(timezone.utc)
        report = SimulationReport(
            runs=[], judge_target=JUDGE, repeat=1, started_at=now, finished_at=now
        )
        assert report.state == "passed"
        assert report.exit_code == 0
        assert report.to_dict()["cases"] == 0


class TestImportOrder:
    """``vision_agents.core`` and ``vision_agents.testing`` import each other; both orders must work."""

    @pytest.mark.parametrize(
        "statement",
        [
            "import vision_agents.testing",
            "import vision_agents.core; import vision_agents.testing",
            "from vision_agents.core import Runner",
        ],
    )
    def test_fresh_interpreter_imports(self, statement: str):
        result = subprocess.run(
            [sys.executable, "-c", statement], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr


class TestParseVerdict:
    def test_pass_and_fail(self):
        assert parse_verdict('{"verdict": "PASS", "reason": "ok"}').success is True
        verdict = parse_verdict('{"verdict": "fail", "reason": "nope"}')
        assert verdict.success is False
        assert verdict.reason == "nope"

    def test_strips_code_fences_and_defaults_reason(self):
        verdict = parse_verdict('```json\n{"verdict": "pass"}\n```')
        assert verdict.success is True
        assert verdict.reason == "Passed."

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("not json", "Could not parse JSON"),
            ('["pass"]', "Expected a JSON object"),
            ('{"verdict": "maybe"}', "Unknown verdict 'maybe'"),
        ],
    )
    def test_rejects_non_verdicts(self, text: str, message: str):
        with pytest.raises(ValueError, match=message):
            parse_verdict(text)
