"""``vision-agents init`` — scaffold a new agent project."""

import logging
import tempfile
from pathlib import Path

import click
from jinja2 import TemplateError

from vision_agents.cli.init.scaffold import scaffold
from vision_agents.cli.uv import install_dependencies

logger = logging.getLogger(__name__)


@click.command("init", help="Scaffold a new agent project from a template.")
@click.argument("name")
@click.option(
    "--no-install",
    is_flag=True,
    help="Do not run 'uv sync' after generating the project.",
)
def init_cmd(name: str, no_install: bool) -> None:
    install = not no_install
    target = Path(name).resolve()
    if target.exists():
        raise click.ClickException(f"{target} already exists")

    # Scaffold into a staging dir on the same filesystem, then rename
    # atomically so partial output is never observable at `target` and
    # we never delete a path we didn't create.
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_ctx = tempfile.TemporaryDirectory(
            prefix=f".{target.name}-init-", dir=target.parent
        )
    except OSError as err:
        raise click.ClickException(
            f"failed to prepare scaffold target {target}: {err}"
        ) from err

    with tmp_ctx as tmp:
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

    if install:
        install_dependencies(target)

    click.echo("\nNext steps:")
    click.echo(f"  cd {name}")
    click.echo("  Copy .env.example to .env and fill in keys")
    if not install:
        click.echo("  uv sync")
    click.echo("  uv run vision-agents agent run")
