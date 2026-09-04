"""Jinja template rendering for ``vision-agents init``."""

import tempfile
from pathlib import Path

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape

TEMPLATE_FILES: dict[str, str] = {
    "agent.py.j2": "agent.py",
    "tests/test_agent.py.j2": "tests/test_agent.py",
    "pyproject.toml.j2": "pyproject.toml",
    "env.example.j2": ".env.example",
    "gitignore.j2": ".gitignore",
    "dockerignore.j2": ".dockerignore",
    "Dockerfile.j2": "Dockerfile",
    "README.md.j2": "README.md",
}


def jinja_env() -> Environment:
    return Environment(
        loader=PackageLoader("vision_agents.cli.init", "templates"),
        autoescape=select_autoescape(default=False),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )


def create_project(name: str, target: Path) -> None:
    """Create a new agent project at ``target`` named ``name``.

    Renders into a sibling staging dir on the same filesystem, then renames
    atomically so partial output is never observable at ``target`` and we
    never delete a path we didn't create.

    Raises:
        FileExistsError: if ``target`` already exists.
        OSError: on filesystem failures.
        jinja2.TemplateError: on template rendering failures.
    """
    if target.exists():
        raise FileExistsError(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{target.name}-init-", dir=target.parent
    ) as tmp:
        staging = Path(tmp) / target.name
        _render_templates(name, staging)
        staging.rename(target)


def _render_templates(project_name: str, target: Path) -> None:
    target.mkdir(parents=True)
    env = jinja_env()
    context = {"project_name": project_name}
    for src, dst in TEMPLATE_FILES.items():
        rendered = env.get_template(src).render(**context)
        out = target / dst
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered, encoding="utf-8")
