"""Jinja template rendering for ``vision-agents init``."""

from pathlib import Path

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape

TEMPLATE_FILES: dict[str, str] = {
    "agent.py.j2": "agent.py",
    "pyproject.toml.j2": "pyproject.toml",
    "env.example.j2": ".env.example",
    "gitignore.j2": ".gitignore",
    "README.md.j2": "README.md",
}


def jinja_env() -> Environment:
    return Environment(
        loader=PackageLoader("vision_agents.cli.init", "templates"),
        autoescape=select_autoescape(default=False),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )


def scaffold(project_name: str, target: Path) -> None:
    target.mkdir(parents=True)
    env = jinja_env()
    context = {"project_name": project_name}
    for src, dst in TEMPLATE_FILES.items():
        rendered = env.get_template(src).render(**context)
        (target / dst).write_text(rendered, encoding="utf-8")
