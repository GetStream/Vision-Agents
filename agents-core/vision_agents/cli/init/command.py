"""``vision-agents init`` — scaffold a new agent project."""

import logging
import subprocess
import tempfile
from pathlib import Path

import click
from jinja2 import TemplateError

from vision_agents.cli.init.scaffold import scaffold
from vision_agents.cli.init.uv import run_uv_sync

logger = logging.getLogger(__name__)


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

    # Scaffold into a staging dir on the same filesystem, then rename
    # atomically so partial output is never observable at `target` and
    # we never delete a path we didn't create.
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{target.name}-init-", dir=target.parent
    ) as tmp:
        staging = Path(tmp) / target.name
        try:
            scaffold(target.name, staging)
        except (OSError, TemplateError) as err:
            raise click.ClickException(f"failed to scaffold {target}: {err}") from err
        try:
            staging.rename(target)
        except OSError as err:
            raise click.ClickException(f"failed to finalize {target}: {err}") from err
    click.echo(f"Created {target}")

    installed = False
    if not no_install:
        try:
            installed = run_uv_sync(target)
        except subprocess.CalledProcessError as err:
            raise click.ClickException(
                f"'uv sync' failed with exit code {err.returncode}"
            ) from err
        if not installed:
            click.echo("warning: 'uv' not found in PATH; skipping venv setup", err=True)

    click.echo("\nNext steps:")
    click.echo(f"  cd {name}")
    click.echo("  Copy .env.example to .env and fill in keys")
    if not installed:
        click.echo("  uv sync")
    click.echo("  vision-agents app run")
