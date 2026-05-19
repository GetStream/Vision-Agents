"""Data types shared between the agent CLI submodules."""

from pathlib import Path
from typing import NamedTuple


class ResolvedEntrypoint(NamedTuple):
    """A fully resolved agent entrypoint ready for in-process dispatch."""

    project_root: Path
    module: str
    attribute: str
    hint: str  # user-facing hint pointing at the source of this resolution
