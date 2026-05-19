"""Import a ``module:attribute`` target and dispatch its ``.cli()`` in-process."""

import difflib
import importlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from vision_agents.cli.agent.models import ResolvedEntrypoint
from vision_agents.cli.errors import CliError
from vision_agents.core import Runner


@contextmanager
def project_on_sys_path(project_root: Path) -> Iterator[None]:
    """Prepend ``project_root`` to ``sys.path`` for the duration."""
    path_str = str(project_root)
    sys.path.insert(0, path_str)
    try:
        yield
    finally:
        try:
            sys.path.remove(path_str)
        except ValueError:
            pass


def dispatch_target(target: ResolvedEntrypoint, args: tuple[str, ...]) -> None:
    """Import ``target.module``, resolve ``target.attribute``, invoke its ``.cli()``.

    Raises:
        CliError: with a clear, hint-augmented message on import,
            attribute-resolution, or dispatch failures.
    """
    with project_on_sys_path(target.project_root):
        try:
            module = importlib.import_module(target.module)
        except ImportError as err:
            raise CliError(
                f"failed to import module '{target.module}': {err}\n{target.hint}"
            ) from err

        attr: object = getattr(module, target.attribute, None)
        if attr is None:
            available = [n for n in dir(module) if not n.startswith("_")]
            suggestions = difflib.get_close_matches(target.attribute, available, n=1)
            did_you_mean = (
                f" Did you mean '{target.module}:{suggestions[0]}'?"
                if suggestions
                else ""
            )
            raise CliError(
                f"module '{target.module}' has no attribute '{target.attribute}'."
                f"{did_you_mean}\n{target.hint}"
            )

        if not isinstance(attr, Runner):
            raise CliError(
                f"'{target.module}:{target.attribute}' is not a Runner instance "
                f"(got {type(attr).__name__}).\n{target.hint}"
            )

        runner: Runner = attr
        runner.cli(args=list(args))
