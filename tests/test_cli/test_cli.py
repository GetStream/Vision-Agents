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
            "pyproject.toml",
            ".env.example",
            ".gitignore",
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
