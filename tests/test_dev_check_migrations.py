import subprocess
import sys
from pathlib import Path

import pytest

DEV_PY = Path(__file__).resolve().parent.parent / "dev.py"
MIGRATIONS = Path("acceleration/migrations")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _commit(repo: Path, *names: str) -> None:
    for name in names:
        (repo / MIGRATIONS / name).write_text("-- +goose Up\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "migrations")


def _check(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(DEV_PY), "check-migrations", "--base", "base"],
        cwd=repo,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repo whose `base` branch has two migrations, with `feature` checked out from it."""
    (tmp_path / MIGRATIONS).mkdir(parents=True)
    _git(tmp_path, "init", "-b", "base")
    _commit(tmp_path, "20261215090000_a.sql", "20261215120000_b.sql")
    _git(tmp_path, "checkout", "-b", "feature")
    return tmp_path


class TestCheckMigrations:
    def test_fails_on_stale_versions_and_suggests_slots_in_order(self, repo: Path):
        _commit(repo, "20261214100000_old.sql", "20261215110000_older_than_b.sql")

        result = _check(repo)

        assert result.returncode == 1
        assert "Newest migration on base: 20261215120000." in result.stderr
        assert "20261214100000_old.sql: version 20261214100000 is not after" in (
            result.stderr
        )
        assert "Rename it to 20261215130000_old.sql." in result.stderr
        assert "Rename it to 20261215140000_older_than_b.sql." in result.stderr

    def test_passes_when_versions_follow_base(self, repo: Path):
        _commit(repo, "20261216000000_new.sql")

        result = _check(repo)

        assert result.returncode == 0, result.stderr
        assert "follow base (newest there: 20261215120000)" in result.stdout

    def test_compares_with_base_tip_not_the_branch_point(self, repo: Path):
        _commit(repo, "20261216000000_mine.sql")
        _git(repo, "checkout", "base")
        _commit(repo, "20261217000000_theirs.sql")
        _git(repo, "checkout", "feature")

        result = _check(repo)

        assert result.returncode == 1
        assert "Rename it to 20261217010000_mine.sql." in result.stderr

    def test_catches_a_migration_renamed_to_a_stale_version(self, repo: Path):
        _git(
            repo,
            "mv",
            str(MIGRATIONS / "20261215120000_b.sql"),
            str(MIGRATIONS / "20261201000000_b.sql"),
        )
        _git(repo, "commit", "-m", "rename")

        result = _check(repo)

        assert result.returncode == 1
        assert "20261201000000_b.sql: version 20261201000000 is not after" in (
            result.stderr
        )

    def test_fails_when_two_added_migrations_share_a_version(self, repo: Path):
        _commit(repo, "20261216000000_one.sql", "20261216000000_two.sql")

        result = _check(repo)

        assert result.returncode == 1
        assert "20261216000000_two.sql: version 20261216000000 is taken" in (
            result.stderr
        )
        assert "Rename it to 20261216010000_two.sql." in result.stderr
        assert "_one.sql:" not in result.stderr
