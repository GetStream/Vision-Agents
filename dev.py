#!/usr/bin/env python3
"""
Development CLI tool for agents-core
Essential dev commands for testing, linting, and type checking
"""

import os
import shlex
import subprocess
import sys
from typing import Optional

import click


def run(
    command: str, env: Optional[dict] = None, check: bool = True
) -> subprocess.CompletedProcess:
    """Run a shell command with automatic argument parsing."""
    click.echo(f"Running: {command}")

    # Set up environment
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    try:
        cmd_list = shlex.split(command)
        result = subprocess.run(
            cmd_list, check=check, capture_output=False, env=full_env, text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        if check:
            click.echo(f"Command failed with exit code {e.returncode}", err=True)
            sys.exit(e.returncode)
        return e


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Development CLI tool for agents-core."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
def test_integration():
    """Run integration tests (requires secrets in place)."""
    click.echo("Running integration tests...")
    run("uv run py.test -m integration")


@cli.command()
def test():
    """Run all tests except integration tests."""
    click.echo("Running unit tests...")
    run("uv run py.test -m 'not integration'")


@cli.command()
def test_plugins():
    """Run plugin tests (TODO: not quite right. uv env is different for each plugin)."""
    click.echo("Running plugin tests...")
    run("uv run py.test plugins/*/tests/*.py -m 'not integration'")


@cli.command()
def format():
    """Run ruff formatting with auto-fix."""
    click.echo("Running ruff format...")
    run("uv run ruff check --fix")


@cli.command()
def lint():
    """Run ruff linting (check only)."""
    click.echo("Running ruff lint...")
    run("uv run ruff format --check .")


@cli.command()
def mypy():
    """Run mypy type checks on main package."""
    click.echo("Running mypy on vision_agents...")
    run("uv run mypy --install-types --non-interactive -p vision_agents")


@cli.command()
def mypy_plugins():
    """Run mypy type checks on all plugins."""
    click.echo("Running mypy on plugins...")
    run(
        "uv run mypy --install-types --non-interactive --exclude 'plugins/.*/tests/.*' plugins"
    )


@cli.command()
def check():
    """Run full check: ruff, mypy, and unit tests."""
    click.echo("Running full development check...")

    # Run ruff
    click.echo("\n=== 1. Ruff Linting ===")
    run("uv run ruff format")
    run("uv run ruff format --check .")

    # Run mypy on main package
    click.echo("\n=== 2. MyPy Type Checking ===")
    run("uv run mypy --install-types --non-interactive -p vision_agents")

    # Run mypy on plugins
    click.echo("\n=== 3. MyPy Plugin Type Checking ===")
    run(
        "uv run mypy --install-types --non-interactive --exclude 'plugins/.*/tests/.*' plugins"
    )

    # Run unit tests
    click.echo("\n=== 4. Unit Tests ===")
    run("uv run py.test -m 'not integration' -n auto")

    click.echo("\n✅ All checks passed!")


if __name__ == "__main__":
    cli()
