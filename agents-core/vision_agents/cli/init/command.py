"""``vision-agents init`` — scaffold a new agent project."""

from pathlib import Path

import click
from jinja2 import TemplateError

from vision_agents.cli.errors import CliError
from vision_agents.cli.init.scaffold import create_project
from vision_agents.cli.init.uv import install_dependencies


@click.command(
    "init",
    help=(
        "Create a new agent project.\n\n"
        "\b\n"
        "Arguments:\n"
        "  AGENT_NAME  Name for the new agent. Used as the directory and\n"
        "              Python project name, e.g. my-agent."
    ),
)
@click.argument("name", required=False, metavar="AGENT_NAME")
@click.option(
    "--no-install",
    is_flag=True,
    help="Do not run 'uv sync' after generating the project.",
)
def init_cmd(name: str | None, no_install: bool) -> None:
    if not name:
        raise CliError(
            "agent name is required.\n\nExample: vision-agents init my-agent"
        )
    install = not no_install
    target = Path(name).resolve()

    try:
        create_project(target.name, target)
    except FileExistsError:
        raise CliError(f"{target} already exists") from None
    except (OSError, TemplateError) as err:
        raise CliError(f"failed to create project at {target}: {err}") from err

    click.echo(f"{click.style('Created', fg='green', bold=True)} {target}")

    if install:
        click.echo(click.style("Installing dependencies (uv sync)...", dim=True))
        install_dependencies(target)

    click.echo()
    click.secho("Next steps:", bold=True)
    click.echo(click.style(f"  cd {name}", fg="cyan"))
    click.echo(
        f"  Copy {click.style('.env.example', fg='cyan')} "
        f"to {click.style('.env', fg='cyan')} and fill in keys"
    )
    if not install:
        click.echo(click.style("  uv sync", fg="cyan"))
    click.echo(click.style("  uv run agent.py run", fg="cyan"))
