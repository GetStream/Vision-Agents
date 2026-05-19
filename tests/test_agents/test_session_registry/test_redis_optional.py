"""Behavior of the Redis optional extra when the `redis` package is absent.

Both tests run in a subprocess that blocks ``import redis`` so they don't
depend on whether the dev venv has the redis package installed.
"""

import subprocess
import sys


def _run_with_redis_blocked(snippet: str) -> subprocess.CompletedProcess[str]:
    code = "import sys\nsys.modules['redis'] = None\n" + snippet
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )


def test_importing_vision_agents_core_is_silent_when_redis_missing() -> None:
    result = _run_with_redis_blocked(
        "import warnings\n"
        "with warnings.catch_warnings(record=True) as w:\n"
        "    warnings.simplefilter('always')\n"
        "    import vision_agents.core  # noqa: F401\n"
        "print(len(w))\n"
        "for entry in w:\n"
        "    print(repr(entry.message))\n"
    )
    assert result.returncode == 0, result.stderr
    first_line = result.stdout.splitlines()[0]
    assert first_line == "0", (
        f"expected zero warnings from `import vision_agents.core`, got:\n{result.stdout}"
    )


def test_direct_redis_store_import_raises_actionable_error() -> None:
    result = _run_with_redis_blocked(
        "try:\n"
        "    import vision_agents.core.agents.session_registry.redis_store  # noqa: F401\n"
        "except ModuleNotFoundError as exc:\n"
        "    print(exc.name)\n"
        "    print(exc)\n"
        "else:\n"
        "    raise SystemExit('expected ModuleNotFoundError')\n"
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == "redis"
    message = "\n".join(lines[1:])
    assert "vision-agents[redis]" in message
