"""Register desktop-control tools on any LLM."""

import logging

from vision_agents.core.llm.llm import LLM

from . import _actions
from ._grid import Grid

logger = logging.getLogger(__name__)


def register(
    llm: LLM,
    grid: Grid | None = None,
    cols: int = 15,
    rows: int = 15,
) -> None:
    """Register all computer-use action tools on *llm*.

    Args:
        llm: The LLM to register tools on.
        grid: Shared Grid instance. If provided, cols/rows are ignored.
        cols: Number of grid columns (1-26). Default 15.
        rows: Number of grid rows (1-99). Default 15.
    """
    if grid is None:
        grid = Grid(cols=cols, rows=rows)

    cell_hint = (
        f" The screen has a grid overlay with columns"
        f" {grid.col_labels[0]}-{grid.col_labels[-1]}"
        f" and rows 1-{grid.rows}."
        " Provide the 'cell' parameter (e.g. 'H8') to target a grid cell."
        " You MUST also provide 'position' to specify where within the cell"
        " (top-left, top, top-right, left, center, right,"
        " bottom-left, bottom, bottom-right)."
    )

    descriptions = {
        "click": "Click at a grid cell." + cell_hint,
        "double_click": "Double-click at a grid cell." + cell_hint,
        "type_text": "Type a string of text into the currently focused element.",
        "key_press": 'Press a key or key combination, e.g. "cmd+c", "enter", "ctrl+shift+t".',
        "scroll": "Scroll at a grid cell." + cell_hint,
        "mouse_move": "Move the mouse cursor to a grid cell." + cell_hint,
        "open_path": "Open a file or folder by its absolute path using the OS default handler.",
    }

    grid_actions = _actions.make_grid_actions(grid)
    tools: dict[str, _actions.ActionFunc] = {
        **grid_actions,
        "type_text": _actions.type_text,
        "key_press": _actions.key_press,
        "open_path": _actions.open_path,
    }

    for name, func in tools.items():
        llm.function_registry.register(name=name, description=descriptions[name])(func)
    logger.info(
        "Registered %d computer-use tools on %s (grid %s)",
        len(tools),
        type(llm).__name__,
        grid.label,
    )
