import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from vision_agents.cli.agent import agent_cmd
from vision_agents.cli.init import init_cmd


class TestInitCommand:
    @pytest.fixture
    def runner(self) -> CliRunner:
        return CliRunner()

    def test_scaffolds_all_expected_files(self, runner: CliRunner, tmp_path: Path):
        target = tmp_path / "my-bot"
        result = runner.invoke(init_cmd, [str(target), "--no-install"])
        assert result.exit_code == 0, result.output
        for name in (
            "agent.py",
            "tests/test_agent.py",
            "pyproject.toml",
            ".env.example",
            ".gitignore",
            ".dockerignore",
            "Dockerfile",
            "README.md",
        ):
            assert (target / name).is_file(), f"missing {name}"

    def test_project_name_uses_target_basename(self, runner: CliRunner, tmp_path: Path):
        target = tmp_path / "my-bot"
        result = runner.invoke(init_cmd, [str(target), "--no-install"])
        assert result.exit_code == 0, result.output
        pyproject = (target / "pyproject.toml").read_text()
        assert 'name = "my-bot"' in pyproject

    def test_errors_if_target_exists(self, runner: CliRunner, tmp_path: Path):
        target = tmp_path / "my-bot"
        target.mkdir()
        result = runner.invoke(init_cmd, [str(target), "--no-install"])
        assert result.exit_code != 0
        assert "already exists" in result.output

    def test_errors_with_friendly_message_when_name_missing(self, runner: CliRunner):
        result = runner.invoke(init_cmd, [])
        assert result.exit_code != 0
        assert "agent name is required" in result.output
        assert "vision-agents init my-agent" in result.output


class TestAgentCommand:
    @pytest.fixture
    def runner(self) -> CliRunner:
        return CliRunner()

    def test_errors_outside_project(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(agent_cmd, [])
        assert result.exit_code != 0
        assert "Could not find pyproject.toml" in result.output

    def test_errors_when_agent_section_missing(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(agent_cmd, [])
        assert result.exit_code != 0
        assert "[tool.vision-agents.agent]" in result.output

    def test_errors_when_entrypoint_is_not_module_attribute_form(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        (tmp_path / "pyproject.toml").write_text(
            '[tool.vision-agents.agent]\nentrypoint = "agent.py"\n'
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(agent_cmd, [])
        assert result.exit_code != 0
        assert "module:attribute" in result.output

    def test_errors_when_entrypoint_has_multiple_colons(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        (tmp_path / "pyproject.toml").write_text(
            '[tool.vision-agents.agent]\nentrypoint = "agent:runner:extra"\n'
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(agent_cmd, [])
        assert result.exit_code != 0
        assert "exactly one ':'" in result.output

    def test_errors_when_entrypoint_has_py_suffix(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        (tmp_path / "pyproject.toml").write_text(
            '[tool.vision-agents.agent]\nentrypoint = "agent.py:runner"\n'
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(agent_cmd, [])
        assert result.exit_code != 0
        assert "looks like a file path" in result.output
        assert "agent:runner" in result.output

    def test_errors_when_module_not_importable(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        (tmp_path / "pyproject.toml").write_text(
            '[tool.vision-agents.agent]\nentrypoint = "no_such_mod:runner"\n'
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(agent_cmd, [])
        assert result.exit_code != 0
        assert "failed to import" in result.output

    def test_errors_when_attribute_missing_in_module(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        (tmp_path / "pyproject.toml").write_text(
            '[tool.vision-agents.agent]\nentrypoint = "stub_attr_missing:runner"\n'
        )
        (tmp_path / "stub_attr_missing.py").write_text("present = 1\n")
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(agent_cmd, [])
        assert result.exit_code != 0
        assert "has no attribute 'runner'" in result.output
        assert "tool.vision-agents.agent.entrypoint" in result.output
        assert str(tmp_path / "pyproject.toml") in result.output

    def test_attribute_missing_suggests_close_match(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        (tmp_path / "pyproject.toml").write_text(
            '[tool.vision-agents.agent]\nentrypoint = "stub_typo:runneraa"\n'
        )
        (tmp_path / "stub_typo.py").write_text("runner = 1\n")
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(agent_cmd, [])
        assert result.exit_code != 0
        assert "Did you mean 'stub_typo:runner'?" in result.output

    def test_errors_when_vision_agents_section_is_not_a_table(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        (tmp_path / "pyproject.toml").write_text('[tool]\nvision-agents = "oops"\n')
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(agent_cmd, [])
        assert result.exit_code != 0
        assert "[tool.vision-agents.agent]" in result.output

    def test_dispatches_to_dotted_module_entrypoint(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        log_file = tmp_path / "called.txt"
        (tmp_path / "pyproject.toml").write_text(
            '[tool.vision-agents.agent]\nentrypoint = "pkg.api:runner"\n'
        )
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "api.py").write_text(
            "from vision_agents.core import Runner\n"
            "class _Runner(Runner):\n"
            "    def __init__(self): pass\n"
            "    def cli(self, args=None):\n"
            f"        open({str(log_file)!r}, 'w').write('ok')\n"
            "runner = _Runner()\n"
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(agent_cmd, [])
        assert result.exit_code == 0, result.output
        assert log_file.read_text() == "ok"

    def test_dispatches_to_runner_cli_with_forwarded_args(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        log_file = tmp_path / "called.txt"
        (tmp_path / "pyproject.toml").write_text(
            '[tool.vision-agents.agent]\nentrypoint = "stub_runner:runner"\n'
        )
        (tmp_path / "stub_runner.py").write_text(
            "from vision_agents.core import Runner\n"
            "class _Runner(Runner):\n"
            "    def __init__(self): pass\n"
            "    def cli(self, args=None):\n"
            f"        open({str(log_file)!r}, 'w').write(' '.join(args or []))\n"
            "runner = _Runner()\n"
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(agent_cmd, ["run", "--debug"])
        assert result.exit_code == 0, result.output
        assert log_file.read_text() == "run --debug"

    def test_errors_when_target_is_not_a_runner_instance(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        (tmp_path / "pyproject.toml").write_text(
            '[tool.vision-agents.agent]\nentrypoint = "stub_not_runner:thing"\n'
        )
        (tmp_path / "stub_not_runner.py").write_text(
            "class Thing:\n    def cli(self): pass\nthing = Thing()\n"
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(agent_cmd, [])
        assert result.exit_code != 0
        assert "is not a Runner instance" in result.output
        assert "got Thing" in result.output

    def test_entrypoint_flag_overrides_config(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        log_file = tmp_path / "called.txt"
        (tmp_path / "pyproject.toml").write_text(
            '[tool.vision-agents.agent]\nentrypoint = "wrong:runner"\n'
        )
        (tmp_path / "override_mod.py").write_text(
            "import sys\n"
            "from vision_agents.core import Runner\n"
            "class _Runner(Runner):\n"
            "    def __init__(self): pass\n"
            "    def cli(self, args=None):\n"
            f"        open({str(log_file)!r}, 'w').write('called')\n"
            "alt = _Runner()\n"
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(agent_cmd, ["--entrypoint=override_mod:alt", "run"])
        assert result.exit_code == 0, result.output
        assert log_file.read_text() == "called"

    def test_entrypoint_flag_works_without_config(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        log_file = tmp_path / "called.txt"
        (tmp_path / "noconfig_mod.py").write_text(
            "from vision_agents.core import Runner\n"
            "class _Runner(Runner):\n"
            "    def __init__(self): pass\n"
            "    def cli(self, args=None):\n"
            f"        open({str(log_file)!r}, 'w').write('ok')\n"
            "runner = _Runner()\n"
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(agent_cmd, ["--entrypoint=noconfig_mod:runner"])
        assert result.exit_code == 0, result.output
        assert log_file.read_text() == "ok"

    def test_entrypoint_flag_rejects_malformed_value(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(agent_cmd, ["--entrypoint=agent.py"])
        assert result.exit_code != 0
        assert "--entrypoint" in result.output
        assert "module:attribute" in result.output


_SIMULATE_STUB = '''
import json
from typing import Optional

from getstream.video.rtc import AudioStreamTrack
from vision_agents.core import Agent, AgentLauncher, Runner, User
from vision_agents.core.edge.edge_transport import EdgeTransport
from vision_agents.core.llm.llm import LLM, LLMResponseFinal
from vision_agents.core.tts import TTS

AGENT_INSTRUCTIONS = "You are the stub weather agent."


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
    """Plays the agent, the caller, the judge or the rewriter, based on the prompt."""

    def __init__(self, error: Optional[str] = None):
        super().__init__()
        self._error = error
        self._caller_turns = 0

    async def simple_response(self, text, participant=None):
        if self._error:
            raise RuntimeError(self._error)
        yield LLMResponseFinal(text=self._reply(text))

    def _reply(self, text):
        if self._instructions == AGENT_INSTRUCTIONS:
            return "It is sunny in Amsterdam today."
        if "Criterion:" in text:
            verdict = "fail" if "impossible" in text else "pass"
            return json.dumps({"verdict": verdict, "reason": "stub judge"})
        if "Brief:" in text:
            return json.dumps([f"Rewrite {i}" for i in range(1, 6)])
        self._caller_turns += 1
        if self._caller_turns == 1:
            return "What's the weather in Amsterdam?"
        return "Thanks, bye! [END]"


def judge():
    return ScriptedLLM()


def broken_judge():
    return ScriptedLLM(error="judge exploded")


def not_an_llm():
    return object()


def _agent(llm):
    return Agent(
        edge=StubEdge(),
        agent_user=User(name="stub", id="stub"),
        instructions=AGENT_INSTRUCTIONS,
        llm=llm,
        tts=StubTTS(),
    )


async def create_agent(**kwargs):
    return _agent(ScriptedLLM())


async def create_broken_agent(**kwargs):
    return _agent(ScriptedLLM(error="provider exploded"))


async def join_call(agent, call_type, call_id, **kwargs):
    pass


runner = Runner(AgentLauncher(create_agent=create_agent, join_call=join_call))
broken_runner = Runner(
    AgentLauncher(create_agent=create_broken_agent, join_call=join_call)
)
'''

_GREETING_SCENARIO = """
name = "greeting"
scenario = "You want to know today's weather in Amsterdam."
criteria = ["The agent tells the user the weather in Amsterdam"]
max_turns = 4
"""

_IMPOSSIBLE_SCENARIO = """
scenario = "You want to know today's weather in Amsterdam."
criteria = [
  "The agent tells the user the weather in Amsterdam",
  "The agent does something impossible",
]
"""


class TestSimulateCommand:
    @pytest.fixture
    def runner(self) -> CliRunner:
        return CliRunner()

    @pytest.fixture
    def project(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.delitem(sys.modules, "sim_stub", raising=False)
        (tmp_path / "pyproject.toml").write_text(
            '[tool.vision-agents.agent]\nentrypoint = "sim_stub:runner"\n'
        )
        (tmp_path / "sim_stub.py").write_text(_SIMULATE_STUB)
        scenarios = tmp_path / "scenarios"
        scenarios.mkdir()
        (scenarios / "greeting.toml").write_text(_GREETING_SCENARIO)
        (scenarios / "impossible.toml").write_text(_IMPOSSIBLE_SCENARIO)
        monkeypatch.chdir(tmp_path)
        return tmp_path

    def _report(self, project: Path, report_dir: str = "simulation-report") -> dict:
        return json.loads((project / report_dir / "report.json").read_text())

    def test_help_describes_options(self, runner: CliRunner, project: Path):
        result = runner.invoke(agent_cmd, ["simulate", "--help"])
        assert result.exit_code == 0, result.output
        for option in ("--repeat", "--variations", "--judge", "--report", "--filter"):
            assert option in result.output
        assert "SCENARIOS" in result.output

    def test_exits_zero_and_prints_table_when_all_pass(
        self, runner: CliRunner, project: Path
    ):
        result = runner.invoke(
            agent_cmd,
            ["simulate", "scenarios/greeting.toml", "--judge", "sim_stub:judge"],
        )
        assert result.exit_code == 0, result.output
        for header in (
            "Scenario",
            "Variations",
            "pass@k",
            "Turns",
            "P50 latency",
            "Failed criteria",
        ):
            assert header in result.output
        assert "greeting" in result.output
        assert "1/1" in result.output
        assert " ms" in result.output

    def test_exits_one_when_a_scenario_fails(self, runner: CliRunner, project: Path):
        result = runner.invoke(
            agent_cmd, ["simulate", "scenarios", "--judge", "sim_stub:judge"]
        )
        assert result.exit_code == 1, result.output
        assert "0/1" in result.output
        assert "The agent does something impossible" in result.output
        report = self._report(project)
        assert report["state"] == "failed"
        assert [run["state"] for run in report["runs"]] == ["passed", "failed"]

    def test_exits_two_when_agent_provider_errors(
        self, runner: CliRunner, project: Path
    ):
        result = runner.invoke(
            agent_cmd,
            [
                "--entrypoint=sim_stub:broken_runner",
                "simulate",
                "scenarios/greeting.toml",
                "--judge",
                "sim_stub:judge",
            ],
        )
        assert result.exit_code == 2, result.output
        report = self._report(project)
        assert report["state"] == "errored"
        case = report["runs"][0]["conversations"][0]
        assert case["state"] == "errored"
        assert case["passed"] is None
        assert "provider exploded" in case["error"]

    def test_exits_two_when_judge_errors(self, runner: CliRunner, project: Path):
        result = runner.invoke(
            agent_cmd,
            ["simulate", "scenarios/greeting.toml", "--judge", "sim_stub:broken_judge"],
        )
        assert result.exit_code == 2, result.output
        case = self._report(project)["runs"][0]["conversations"][0]
        assert case["state"] == "errored"
        assert "judge exploded" in case["error"]

    @pytest.mark.parametrize(
        ("judge", "message"),
        [
            ("nonsense", "provider/model"),
            ("no_such_plugin/some-model", "not installed"),
            ("sim_stub:not_an_llm", "expected an LLM instance"),
            ("sim_stub:missing", "no callable"),
        ],
    )
    def test_exits_two_on_bad_judge_spec(
        self, runner: CliRunner, project: Path, judge: str, message: str
    ):
        result = runner.invoke(
            agent_cmd, ["simulate", "scenarios/greeting.toml", "--judge", judge]
        )
        assert result.exit_code == 2, result.output
        assert message in result.output
        assert not (project / "simulation-report").exists()

    def test_repeat_and_variations_are_reflected_in_report(
        self, runner: CliRunner, project: Path
    ):
        result = runner.invoke(
            agent_cmd,
            [
                "simulate",
                "scenarios/greeting.toml",
                "--judge",
                "sim_stub:judge",
                "--repeat",
                "2",
                "--variations",
                "3",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "6/6" in result.output
        report = self._report(project)
        assert report["repeat"] == 2
        run = report["runs"][0]
        assert run["variations"] == 3
        assert run["cases"] == 6
        assert sorted((c["variation"], c["attempt"]) for c in run["conversations"]) == [
            (v, a) for v in range(3) for a in range(2)
        ]
        briefs = {c["variation"]: c["scenario"] for c in run["conversations"]}
        assert briefs[0] == "You want to know today's weather in Amsterdam."
        assert briefs[1] == "Rewrite 1"
        assert briefs[2] == "Rewrite 2"

    def test_filter_selects_scenarios_by_name(self, runner: CliRunner, project: Path):
        result = runner.invoke(
            agent_cmd,
            ["simulate", "scenarios", "--judge", "sim_stub:judge", "--filter", "GREET"],
        )
        assert result.exit_code == 0, result.output
        assert [run["name"] for run in self._report(project)["runs"]] == ["greeting"]

    def test_filter_without_match_exits_two(self, runner: CliRunner, project: Path):
        result = runner.invoke(
            agent_cmd,
            ["simulate", "scenarios", "--judge", "sim_stub:judge", "--filter", "nope"],
        )
        assert result.exit_code == 2, result.output
        assert "no scenarios match" in result.output

    def test_reports_contain_transcripts_and_verdicts(
        self, runner: CliRunner, project: Path
    ):
        result = runner.invoke(
            agent_cmd,
            [
                "simulate",
                "scenarios",
                "--judge",
                "sim_stub:judge",
                "--report",
                "out/reports",
            ],
        )
        assert result.exit_code == 1, result.output
        report = self._report(project, "out/reports")
        greeting, impossible = report["runs"]
        case = greeting["conversations"][0]
        assert [(line["caller"], line["text"]) for line in case["transcript"]] == [
            (True, "What's the weather in Amsterdam?"),
            (False, "It is sunny in Amsterdam today."),
            (True, "Thanks, bye!"),
            (False, "It is sunny in Amsterdam today."),
        ]
        assert case["turns"] == 2
        assert case["ended"] == "complete"
        assert case["passed"] is True
        assert case["criteria"] == [
            {
                "criterion": "The agent tells the user the weather in Amsterdam",
                "passed": True,
                "reason": "stub judge",
            }
        ]
        failed_case = impossible["conversations"][0]
        assert failed_case["passed"] is False
        assert (
            "The agent does something impossible: stub judge" in failed_case["verdict"]
        )

        markdown = (project / "out" / "reports" / "report.md").read_text()
        assert "# Simulation report" in markdown
        assert "| greeting | 1 | 1/1 |" in markdown
        assert "**User:** What's the weather in Amsterdam?" in markdown
        assert "It is sunny in Amsterdam today." in markdown
        assert "PASS: The agent tells the user the weather in Amsterdam" in markdown
        assert "FAIL: The agent does something impossible (stub judge)" in markdown

    def test_missing_target_exits_two(self, runner: CliRunner, project: Path):
        result = runner.invoke(
            agent_cmd, ["simulate", "nowhere", "--judge", "sim_stub:judge"]
        )
        assert result.exit_code == 2, result.output
        assert "does not exist" in result.output

    def test_malformed_scenario_exits_two(self, runner: CliRunner, project: Path):
        (project / "scenarios" / "broken.toml").write_text('scenario = "x"\n')
        result = runner.invoke(
            agent_cmd, ["simulate", "scenarios", "--judge", "sim_stub:judge"]
        )
        assert result.exit_code == 2, result.output
        assert "broken.toml" in result.output
        assert "criteria" in result.output
