"""``uv`` subprocess helpers for ``vision-agents init``."""

import shutil
import subprocess
from pathlib import Path


def run_uv_sync(target: Path) -> bool:
    """Run ``uv sync`` in ``target``.

    Returns:
        ``True`` if uv was found and ran (raises ``CalledProcessError`` on
        non-zero exit); ``False`` if uv is not on ``PATH``.
    """
    if shutil.which("uv") is None:
        return False
    try:
        subprocess.run(["uv", "sync"], cwd=target, check=True)
    except FileNotFoundError:
        return False
    return True
