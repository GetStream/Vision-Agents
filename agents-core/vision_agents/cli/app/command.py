"""``vision-agents app`` — proxy to the project's Runner CLI."""

import logging
import shutil
import subprocess
import sys
from pathlib import Path

import click

from vision_agents.cli.app.config import find_config, load_app_config

logger = logging.getLogger(__name__)


@click.command(
    "app",
    help="Run the project's agent entrypoint (forwards args to its Runner CLI).",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def app_cmd(args: tuple[str, ...]) -> None:
    config_path = find_config(Path.cwd())
    config = load_app_config(config_path)
    project_root = config_path.parent
    entrypoint = (project_root / str(config["entrypoint"])).resolve()
    if not entrypoint.is_file():
        raise click.ClickException(f"entrypoint {entrypoint} not found")

    if shutil.which("uv") is not None:
        cmd = ["uv", "run", "python", str(entrypoint), *args]
    else:
        cmd = [sys.executable, str(entrypoint), *args]

    result = subprocess.run(cmd, cwd=project_root)
    sys.exit(result.returncode)
