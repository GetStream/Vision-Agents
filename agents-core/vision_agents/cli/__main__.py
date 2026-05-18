"""Entry point for the ``vision-agents`` console script."""

import logging
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import click
from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape

logger = logging.getLogger(__name__)

_CONFIG_FILENAME = "vision-agents.toml"

_TEMPLATE_FILES: dict[str, str] = {
    "agent.py.j2": "agent.py",
    "pyproject.toml.j2": "pyproject.toml",
    "env.example.j2": ".env.example",
    "gitignore.j2": ".gitignore",
    "README.md.j2": "README.md",
    "vision-agents.toml.j2": _CONFIG_FILENAME,
}


def _jinja_env() -> Environment:
    return Environment(
        loader=PackageLoader("vision_agents.cli", "templates"),
        autoescape=select_autoescape(default=False),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )


def _scaffold(project_name: str, target: Path) -> None:
    target.mkdir(parents=True)
    env = _jinja_env()
    context = {"project_name": project_name}
    for src, dst in _TEMPLATE_FILES.items():
        rendered = env.get_template(src).render(**context)
        (target / dst).write_text(rendered, encoding="utf-8")


def _run_uv_sync(target: Path) -> bool:
    if shutil.which("uv") is None:
        return False
    subprocess.run(["uv", "sync"], cwd=target, check=True)
    return True


def _find_config(start: Path) -> Path:
    for directory in (start, *start.parents):
        candidate = directory / _CONFIG_FILENAME
        if candidate.is_file():
            return candidate
    raise click.ClickException(
        "Could not find vision-agents configuration; "
        "you may not be in a Vision Agents project."
    )


def _load_app_config(config_path: Path) -> dict[str, object]:
    try:
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as err:
        raise click.ClickException(f"failed to parse {config_path}: {err}") from err

    app = data.get("app")
    if not isinstance(app, dict):
        raise click.ClickException(f"{config_path} is missing required [app] section")

    entrypoint = app.get("entrypoint")
    if not isinstance(entrypoint, str) or not entrypoint:
        raise click.ClickException(
            f"{config_path} requires app.entrypoint to be a non-empty string"
        )

    return {"entrypoint": entrypoint}


@click.group(help="Vision Agents command-line interface.")
def main() -> None:
    """Top-level command group."""


@main.command(help="Scaffold a new agent project from a template.")
@click.argument("name")
@click.option(
    "--no-install",
    is_flag=True,
    help="Do not run 'uv sync' after generating the project.",
)
def init(name: str, no_install: bool) -> None:
    target = Path(name).resolve()
    if target.exists():
        raise click.ClickException(f"{target} already exists")

    _scaffold(name, target)
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
    click.echo("  uv run python agent.py")


@main.command(
    help="Run the project's agent entrypoint (forwards args to its Runner CLI).",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def app(args: tuple[str, ...]) -> None:
    config_path = _find_config(Path.cwd())
    config = _load_app_config(config_path)
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


if __name__ == "__main__":
    sys.exit(main())
