#!/usr/bin/env python3
"""
Development CLI tool for agents-core
Essential dev commands for testing, linting, and type checking
"""

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple, Optional

import click
import setuptools
import toml

CORE_EXTRAS_ALL_SECTION = "all-plugins"
CORE_EXTRAS_DEV_SECTION = "dev"
CORE_PACKAGE_NAME = "agents-core"
PLUGINS_DIR = "plugins"


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


def _plugin_mypypath() -> str:
    """Build MYPYPATH with all plugin root directories."""
    return ":".join(str(p) for p in sorted(Path(PLUGINS_DIR).iterdir()) if p.is_dir())


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
        "uv run mypy --install-types --non-interactive --exclude '/(tests|examples?)/' plugins",
        env={"MYPYPATH": _plugin_mypypath()},
    )


class CoreDependencies(NamedTuple):
    all: list[str]
    plugins: dict[str, list[str]]


def _cwd_is_root():
    cwd = Path.cwd()
    return (cwd / CORE_PACKAGE_NAME).exists() and (cwd / PLUGINS_DIR).exists()


def _get_plugin_package_name(plugin: str) -> str:
    with open(Path(PLUGINS_DIR) / Path(plugin) / "pyproject.toml", "r") as f:
        pyproject = toml.load(f)
    return pyproject["project"]["name"]


def _get_core_optional_dependencies() -> CoreDependencies:
    with open(Path(CORE_PACKAGE_NAME) / "pyproject.toml", "r") as f:
        pyproject = toml.load(f)

    optionals: dict[str, list[str]] = pyproject.get("project", {}).get(
        "optional-dependencies", {}
    )
    optionals_all = optionals.get(CORE_EXTRAS_ALL_SECTION, [])
    optionals_plugins = {
        k: v
        for k, v in optionals.items()
        if k not in (CORE_EXTRAS_ALL_SECTION, CORE_EXTRAS_DEV_SECTION)
    }
    return CoreDependencies(all=optionals_all, plugins=optionals_plugins)


@cli.command(name="validate-extras")
def validate_extra_dependencies():
    """
    Validate that all namespace packages are include into optional dependencies in "agents-core/pyproject.toml".
    This command must be executed from the project root.
    """
    # First, validate that the script is executed from the project's root
    if not _cwd_is_root():
        raise RuntimeError("The script must be executed from the project root.")

    # Get all namespace packages in plugins/
    plugins = setuptools.find_namespace_packages(PLUGINS_DIR)
    plugins_roots = {p.split(".")[0] for p in plugins}
    plugins_packages = [_get_plugin_package_name(plugin) for plugin in plugins_roots]

    # Get optional dependencies for "agents-core" package.
    core_optional_dependencies = _get_core_optional_dependencies()

    # Validate that "agents-core" has "all-plugins" section in optional dependencies
    if not core_optional_dependencies.all:
        raise click.ClickException(
            f'Optional dependencies for "{CORE_PACKAGE_NAME}" are missing the "{CORE_EXTRAS_ALL_SECTION}" section.'
        )

    # Validate that all available plugins are listed in "all-plugins"
    not_included_in_all = set(plugins_packages) - set(core_optional_dependencies.all)
    if not_included_in_all:
        raise click.ClickException(
            f'The following plugins are not included in the "{CORE_EXTRAS_ALL_SECTION}" '
            f'section in "{CORE_PACKAGE_NAME}" package: {", ".join(not_included_in_all)}"'
        )

    # Validate that every plugin has a dedicated section in core's optional dependencies
    plugins_sections_reversed = {
        tuple(v): k for k, v in core_optional_dependencies.plugins.items()
    }
    plugins_without_optional = []
    for package_name in plugins_packages:
        if (package_name,) not in plugins_sections_reversed:
            plugins_without_optional.append(package_name)

    if plugins_without_optional:
        raise click.ClickException(
            f"The following plugins do not have an optional dependency section "
            f'in "{CORE_PACKAGE_NAME}" package: \n{", ".join(plugins_without_optional)}". \n\n'
            f'To fix it, add a section for each plugin to [project.optional-dependencies] inside "{CORE_PACKAGE_NAME}/pyproject.toml" like this: \n\n'
            f'plugin_name = ["vision-agents-plugins-plugin-name"]'
        )
    return None


@cli.command()
def check():
    """Run full check: ruff, mypy, and unit tests."""
    click.echo("Running full development check...")

    # Run ruff
    click.echo("\n=== 1. Ruff Linting ===")
    run("uv run ruff format")
    run("uv run ruff format --check .")

    # Validate extra dependencies included to agents-core/pyproject.toml
    click.echo("\n=== 2. Validate agents-core/pyproject.toml ===")
    validate_extra_dependencies.callback()

    # Run mypy on main package
    click.echo("\n=== 3. MyPy Type Checking ===")
    mypy.callback()

    # Run mypy on plugins
    click.echo("\n=== 4. MyPy Plugin Type Checking ===")
    mypy_plugins.callback()

    # Run unit tests
    click.echo("\n=== 5. Unit Tests ===")
    run("uv run py.test -m 'not integration' -n auto")

    click.echo("\n✅ All checks passed!")


if __name__ == "__main__":
    cli()
