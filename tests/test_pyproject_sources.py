from __future__ import annotations

from pathlib import Path
import tomllib

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def _collect_uv_path_entries() -> list[tuple[Path, str, Path]]:
    """Collect path-based UV source entries from every pyproject.toml in the repo."""
    entries: list[tuple[Path, str, Path]] = []

    for pyproject in REPO_ROOT.rglob("pyproject.toml"):
        if ".venv" in pyproject.parts:
            continue

        with pyproject.open("rb") as handle:
            data = tomllib.load(handle)

        sources = data.get("tool", {}).get("uv", {}).get("sources")
        if not isinstance(sources, dict):
            continue

        for package_name, source_config in sources.items():
            if not isinstance(source_config, dict):
                continue

            raw_path = source_config.get("path")
            if raw_path is None:
                continue

            entries.append((pyproject, package_name, Path(str(raw_path))))

    return entries


UV_PATH_ENTRIES = _collect_uv_path_entries()
UV_PATH_ENTRY_IDS = [
    f"{pyproject_path.relative_to(REPO_ROOT)}::{package_name}"
    for pyproject_path, package_name, _ in UV_PATH_ENTRIES
]


@pytest.mark.skipif(not UV_PATH_ENTRIES, reason="No path-based uv sources found.")
@pytest.mark.parametrize(
    ("pyproject_path", "package_name", "raw_path"),
    UV_PATH_ENTRIES,
    ids=UV_PATH_ENTRY_IDS,
)
def test_uv_source_paths_exist(
    pyproject_path: Path,
    package_name: str,
    raw_path: Path,
) -> None:
    """Ensure each UV path-based source resolves to a real package directory.

    Args:
        pyproject_path: Location of the pyproject file that declares the source.
        package_name: Package name used in the `[tool.uv.sources]` table.
        raw_path: Raw path value specified for the dependency.
    """
    base_dir = pyproject_path.parent
    resolved_path = (base_dir / raw_path).resolve()
    message = (
        f"{pyproject_path.relative_to(REPO_ROOT)}: [tool.uv.sources] entry "
        f"'{package_name}' points to '{raw_path}'"
    )

    try:
        resolved_path.relative_to(REPO_ROOT)
    except ValueError as error:
        raise AssertionError(
            f"{message}, resolved path {resolved_path} is outside the repository",
        ) from error

    assert resolved_path.exists(), (
        f"{message}, resolved path {resolved_path} is missing"
    )
    assert resolved_path.is_dir(), (
        f"{message}, resolved path {resolved_path} is not a directory"
    )
    assert (resolved_path / "pyproject.toml").exists(), (
        f"{message}, pyproject.toml not found at {resolved_path}"
    )
