"""``vision-agents agent`` — resolve the entrypoint, then dispatch in-process."""

from pathlib import Path

import click

from vision_agents.cli.agent.config import resolve_entrypoint
from vision_agents.cli.agent.dispatch import dispatch_target


@click.command(
    "agent",
    help="Dispatch to the project's Runner CLI (forwards remaining args).",
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
        # Everything after the first positional (e.g. `run --help`) belongs to
        # the Runner CLI, so stop parsing our own options there.
        "allow_interspersed_args": False,
    },
)
@click.option(
    "--entrypoint",
    "entrypoint_override",
    default=None,
    metavar="MODULE:ATTR",
    help="Override the agent entrypoint without editing pyproject.toml "
    "(e.g. --entrypoint=agent:runner). Must come before the subcommand.",
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def agent_cmd(entrypoint_override: str | None, args: tuple[str, ...]) -> None:
    target = resolve_entrypoint(Path.cwd(), entrypoint_override)
    dispatch_target(target, args)
