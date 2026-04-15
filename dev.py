#!/usr/bin/env python3
"""
Development CLI tool for agents-core
Essential dev commands for testing, linting, and type checking
"""

import datetime
import json
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
        "uv run mypy --install-types --non-interactive --exclude 'plugins/[^/]+/tests/' --exclude 'plugins/getstream/.*/sfu_events\\.py' --exclude 'plugins/getstream/_generate_sfu_events\\.py' plugins",
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


def _extract_first_failure(
    tests: list[dict],
) -> tuple[str, str, str] | tuple[None, None, None]:
    """
    Return (nodeid, phase, message) for first failed test,
    checking setup → call → teardown.
    """
    for t in tests:
        if t.get("outcome") in ("failed", "error"):
            nodeid = t.get("nodeid", "unknown test")

            for phase in ["setup", "call", "teardown"]:
                phase_data = t.get(phase)
                if phase_data and phase_data.get("outcome") == "failed":
                    longrepr = phase_data.get("longrepr", "")
                    if isinstance(longrepr, dict):
                        message = longrepr.get("reprcrash", {}).get(
                            "message", str(longrepr)
                        )
                    else:
                        message = str(longrepr)

                    return nodeid, phase, message

            return nodeid, "unknown", "No error details available"

    return None, None, None


@cli.command()
@click.option(
    "--file",
    "report_file",
    default=".report.json",
    type=click.File("r"),
    show_default=True,
    help="Path to pytest JSON report file",
)
def parse_pytest_report(report_file):
    """Parse pytest JSON report and print summary for CI (GitHub Actions friendly)."""
    report = json.load(report_file)

    summary = report.get("summary", {})
    duration = str(datetime.timedelta(seconds=report.get("duration", 0)))
    exit_code = report.get("exitcode", 0)

    failed = summary.get("failed", 0)
    error = summary.get("error", 0)
    passed = summary.get("passed", 0)
    skipped = summary.get("skipped", 0)
    total = summary.get("total", 0)

    # Header
    status = (
        "❌ Test Results - Failed" if failed or error else "✅ Test Results - Success"
    )
    click.echo(f"*{status}*")
    click.echo()

    # Duration
    click.echo(f"*Duration:* {duration}")
    click.echo()

    # Summary
    click.echo("*Summary:*")
    click.echo(f"* *Total:* {total}")
    click.echo(f"* *Passed:* {passed}")
    click.echo(f"* *Failed:* {failed}")
    click.echo(f"* *Error:* {error}")
    click.echo(f"* *Skipped:* {skipped}")
    click.echo()

    # Failure details
    if failed or error:
        nodeid, phase, message = _extract_first_failure(report.get("tests", []))

        click.echo("*Failure Details:*")
        click.echo(f"* *Exit code:* `{exit_code}`")

        if nodeid:
            click.echo(f"* *First failed test:* `{nodeid}`")
        if phase:
            click.echo(f"* *Phase:* `{phase}`")
        if message:
            click.echo(f"* *Error:*\n\n```\n{message}\n```")


if __name__ == "__main__":
    cli()
