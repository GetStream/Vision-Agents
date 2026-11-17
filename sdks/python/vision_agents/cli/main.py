"""Top-level ``vision-agents`` command group."""

import click

from vision_agents.cli.agent import agent_cmd
from vision_agents.cli.init import init_cmd


@click.group(help="Vision Agents command-line interface.")
@click.version_option(package_name="vision-agents")
def main() -> None:
    """Top-level command group."""


main.add_command(init_cmd)
main.add_command(agent_cmd)
