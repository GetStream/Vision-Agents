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

    def test_errors_when_entrypoint_missing(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        (tmp_path / "pyproject.toml").write_text(
            '[tool.vision-agents.agent]\nentrypoint = "missing.py"\n'
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(agent_cmd, [])
        assert result.exit_code != 0
        assert "not found" in result.output
