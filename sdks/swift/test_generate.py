"""Behaviour of ``generate.py --check`` and of a run without the generator.

Each test runs the script on a copy of the repository files it reads, so an edit to the spec
or to the generated client stays in ``tmp_path``. Run from the repository root:
``uv run --no-project --with pytest --with pyyaml pytest sdks/swift/test_generate.py``.
"""

import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).parents[2]
GENERATED = Path("sdks/swift/core/Sources/VisionAgentsCore/Generated")
FINGERPRINT = Path("sdks/swift/core/generated.json")
SPEC = Path("acceleration/api/openapi.yaml")
SESSION_MODEL = GENERATED / "Models" / "SessionPayload.swift"


class Checkout:
    """A copy of the files ``generate.py`` reads, laid out as in the repository."""

    def __init__(self, root: Path):
        self.root = root

    def run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(self.root / "sdks/swift/generate.py"), *args],
            capture_output=True,
            text=True,
            check=False,
        )

    def check(self) -> subprocess.CompletedProcess:
        return self.run("--check")

    def edit_spec(self, edit: Callable[[dict], None]) -> None:
        spec = yaml.safe_load((self.root / SPEC).read_text())
        edit(spec)
        (self.root / SPEC).write_text(yaml.dump(spec, sort_keys=False))

    def edit_fingerprint(self, edit: Callable[[dict], None]) -> None:
        path = self.root / FINGERPRINT
        committed = json.loads(path.read_text())
        edit(committed)
        path.write_text(json.dumps(committed))


def add_session_field(spec: dict) -> None:
    spec["components"]["schemas"]["Session"]["properties"]["probe_field"] = {
        "type": "string"
    }


def mark_operation(spec: dict, operation_id: str) -> None:
    for operations in spec["paths"].values():
        for operation in operations.values():
            if operation.get("operationId") == operation_id:
                operation["x-probe"] = True


@pytest.fixture
def checkout(tmp_path: Path) -> Checkout:
    root = tmp_path / "repo"
    (root / SPEC).parent.mkdir(parents=True)
    (root / "sdks/swift").mkdir(parents=True)
    shutil.copy(REPO / SPEC, root / SPEC)
    shutil.copy(REPO / "sdks/swift/generate.py", root / "sdks/swift/generate.py")
    (root / FINGERPRINT).parent.mkdir(parents=True)
    shutil.copy(REPO / FINGERPRINT, root / FINGERPRINT)
    shutil.copytree(REPO / GENERATED, root / GENERATED)
    return Checkout(root)


class TestCheck:
    def test_committed_client_is_up_to_date(self, checkout: Checkout) -> None:
        assert checkout.check().returncode == 0

    def test_spec_rewritten_without_edits_is_up_to_date(
        self, checkout: Checkout
    ) -> None:
        checkout.edit_spec(lambda spec: None)
        assert checkout.check().returncode == 0

    def test_new_field_on_a_client_model_is_stale(self, checkout: Checkout) -> None:
        checkout.edit_spec(add_session_field)
        result = checkout.check()
        assert result.returncode == 1
        assert "stale" in result.stderr

    def test_changed_client_operation_is_stale(self, checkout: Checkout) -> None:
        checkout.edit_spec(lambda spec: mark_operation(spec, "getSession"))
        assert checkout.check().returncode == 1

    def test_server_only_schema_change_is_up_to_date(self, checkout: Checkout) -> None:
        def change(spec: dict) -> None:
            spec["components"]["schemas"]["HealthStatus"]["description"] = "changed"

        checkout.edit_spec(change)
        assert checkout.check().returncode == 0

    def test_server_only_operation_change_is_up_to_date(
        self, checkout: Checkout
    ) -> None:
        checkout.edit_spec(lambda spec: mark_operation(spec, "getHealth"))
        assert checkout.check().returncode == 0

    def test_info_and_servers_change_is_up_to_date(self, checkout: Checkout) -> None:
        def change(spec: dict) -> None:
            spec["info"]["description"] = "changed"
            spec["info"]["title"] = "changed"
            spec["servers"] = [{"url": "http://example.invalid"}]

        checkout.edit_spec(change)
        assert checkout.check().returncode == 0

    def test_missing_fingerprint_is_stale(self, checkout: Checkout) -> None:
        (checkout.root / FINGERPRINT).unlink()
        assert checkout.check().returncode == 1

    def test_fingerprint_of_other_options_is_stale(self, checkout: Checkout) -> None:
        checkout.edit_fingerprint(lambda committed: committed["options"].pop())
        assert checkout.check().returncode == 1

    def test_fingerprint_without_client_hash_is_stale(self, checkout: Checkout) -> None:
        checkout.edit_fingerprint(lambda committed: committed.pop("client_sha256"))
        assert checkout.check().returncode == 1

    def test_hand_edited_generated_file_is_stale(self, checkout: Checkout) -> None:
        path = checkout.root / SESSION_MODEL
        path.write_text(path.read_text() + "\n// hand edit\n")
        result = checkout.check()
        assert result.returncode == 1
        assert "stale" in result.stderr

    def test_generated_file_from_an_older_run_is_stale(
        self, checkout: Checkout
    ) -> None:
        (checkout.root / SESSION_MODEL).write_text("// older run\n")
        assert checkout.check().returncode == 1

    def test_removed_generated_file_is_stale(self, checkout: Checkout) -> None:
        (checkout.root / SESSION_MODEL).unlink()
        assert checkout.check().returncode == 1

    def test_operation_closed_to_clients_fails(self, checkout: Checkout) -> None:
        def close(spec: dict) -> None:
            for operations in spec["paths"].values():
                for operation in operations.values():
                    if operation.get("operationId") == "getSession":
                        del operation["x-client-accessible"]

        checkout.edit_spec(close)
        result = checkout.check()
        assert result.returncode == 1
        assert "getSession is server-side only" in result.stderr


class TestRegenerate:
    def test_without_the_generator_nothing_is_rewritten(
        self, checkout: Checkout
    ) -> None:
        checkout.edit_spec(add_session_field)
        before = (checkout.root / FINGERPRINT).read_text()
        client = (checkout.root / SESSION_MODEL).read_text()

        result = checkout.run()

        assert result.returncode == 1
        assert "no generator" in result.stderr
        assert (checkout.root / FINGERPRINT).read_text() == before
        assert (checkout.root / SESSION_MODEL).read_text() == client
        assert checkout.check().returncode == 1
