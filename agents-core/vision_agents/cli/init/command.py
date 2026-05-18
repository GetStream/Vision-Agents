"""``vision-agents init`` — scaffold a new agent project."""

import logging
import shutil
import subprocess
from pathlib import Path

import click
from jinja2 import TemplateError

from vision_agents.cli.init.scaffold import scaffold

logger = logging.getLogger(__name__)


def _run_uv_sync(target: Path) -> bool:
    if shutil.which("uv") is None:
        return False
    subprocess.run(["uv", "sync"], cwd=target, check=True)
    return True


@click.command("init", help="Scaffold a new agent project from a template.")
@click.argument("name")
@click.option(
    "--no-install",
    is_flag=True,
    help="Do not run 'uv sync' after generating the project.",
)
def init_cmd(name: str, no_install: bool) -> None:
    target = Path(name).resolve()
    if target.exists():
        raise click.ClickException(f"{target} already exists")

    try:
        scaffold(name, target)
    except (OSError, TemplateError) as err:
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        raise click.ClickException(f"failed to scaffold {target}: {err}") from err
    click.echo(f"Created {target}")

    installed = False
    if not no_install:
        try:
            installed = _run_uv_sync(target)
        except subprocess.CalledProcessError as err:
            raise click.ClickException(
                f"'uv sync' failed with exit code {err.returncode}"
            ) from err
        if not installed:
            click.echo("warning: 'uv' not found in PATH; skipping venv setup", err=True)

    click.echo("\nNext steps:")
    click.echo(f"  cd {name}")
    click.echo("  cp .env.example .env  # then fill in keys")
    if not installed:
        click.echo("  uv sync")
    click.echo("  vision-agents app run")
