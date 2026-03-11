"""Low-level desktop actions backed by pyautogui."""

import asyncio
import logging
import platform
import subprocess
from collections.abc import Callable, Coroutine
from typing import Any, Literal

import pyautogui

from ._grid import VIRTUAL_SIZE, Grid

logger = logging.getLogger(__name__)

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.1


def _run_sync(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run a blocking pyautogui call in a thread executor."""
    loop = asyncio.get_running_loop()
    return loop.run_in_executor(None, lambda: func(*args, **kwargs))


def _to_screen(vx: int, vy: int) -> tuple[int, int]:
    """Scale from virtual coordinate space to actual screen pixels."""
    vx = max(0, min(vx, VIRTUAL_SIZE))
    vy = max(0, min(vy, VIRTUAL_SIZE))
    sw, sh = pyautogui.size()
    screen_x = int(vx * sw / VIRTUAL_SIZE)
    screen_y = int(vy * sh / VIRTUAL_SIZE)
    screen_x = max(0, min(screen_x, sw - 1))
    screen_y = max(0, min(screen_y, sh - 1))
    return screen_x, screen_y


ActionFunc = Callable[..., Coroutine[Any, Any, str]]


def make_grid_actions(grid: Grid) -> dict[str, ActionFunc]:
    """Create coordinate-based action functions bound to *grid*."""

    def _cell_to_screen(cell: str, position: str = "center") -> tuple[int, int]:
        vx, vy = grid.cell_to_virtual(cell, position=position)
        return _to_screen(vx, vy)

    async def click(
        cell: str,
        position: str = "center",
        button: str = "left",
    ) -> str:
        """Click at a grid cell.

        Args:
            cell: Grid cell reference, e.g. "H8".
            position: Where within the cell to click. One of: top-left, top,
                top-right, left, center, right, bottom-left, bottom, bottom-right.
            button: Mouse button — "left", "right", or "middle".
        """
        sx, sy = _cell_to_screen(cell, position)
        logger.info(
            "click(cell=%s, position=%s -> screen=%d,%d, button=%s)",
            cell,
            position,
            sx,
            sy,
            button,
        )
        await _run_sync(pyautogui.click, sx, sy, button=button)
        return f"Clicked at {cell} ({position}) with {button} button"

    async def double_click(
        cell: str,
        position: str = "center",
    ) -> str:
        """Double-click at a grid cell.

        Args:
            cell: Grid cell reference, e.g. "H8".
            position: Where within the cell to click. One of: top-left, top,
                top-right, left, center, right, bottom-left, bottom, bottom-right.
        """
        sx, sy = _cell_to_screen(cell, position)
        logger.info(
            "double_click(cell=%s, position=%s -> screen=%d,%d)",
            cell,
            position,
            sx,
            sy,
        )
        await _run_sync(pyautogui.doubleClick, sx, sy)
        return f"Double-clicked at {cell} ({position})"

    async def scroll(
        cell: str,
        position: str = "center",
        clicks: int = 3,
        direction: Literal["up", "down"] = "down",
    ) -> str:
        """Scroll at a grid cell.

        Args:
            cell: Grid cell reference, e.g. "H8".
            position: Where within the cell to scroll. One of: top-left, top,
                top-right, left, center, right, bottom-left, bottom, bottom-right.
            clicks: Number of scroll increments.
            direction: "up" or "down".
        """
        sx, sy = _cell_to_screen(cell, position)
        amount = clicks if direction == "up" else -clicks
        await _run_sync(pyautogui.scroll, amount, x=sx, y=sy)
        logger.debug(
            "scroll(cell=%s, position=%s -> screen=%d,%d, direction=%s)",
            cell,
            position,
            sx,
            sy,
            direction,
        )
        return f"Scrolled {direction} {clicks} clicks at {cell} ({position})"

    async def mouse_move(
        cell: str,
        position: str = "center",
    ) -> str:
        """Move the mouse cursor to a grid cell.

        Args:
            cell: Grid cell reference, e.g. "H8".
            position: Where within the cell to move. One of: top-left, top,
                top-right, left, center, right, bottom-left, bottom, bottom-right.
        """
        sx, sy = _cell_to_screen(cell, position)
        logger.info(
            "mouse_move(cell=%s, position=%s -> screen=%d,%d)",
            cell,
            position,
            sx,
            sy,
        )
        await _run_sync(pyautogui.moveTo, sx, sy)
        return f"Moved mouse to {cell} ({position})"

    return {
        "click": click,
        "double_click": double_click,
        "scroll": scroll,
        "mouse_move": mouse_move,
    }


async def type_text(text: str) -> str:
    """Type a string of text into the currently focused element.

    Args:
        text: The text to type.
    """
    await _run_sync(pyautogui.write, text, interval=0.03)
    logger.debug("type_text(%r)", text[:80])
    return f"Typed {len(text)} characters"


async def key_press(keys: str) -> str:
    """Press a key or key combination.

    Args:
        keys: Key combo separated by "+", e.g. "cmd+c", "ctrl+shift+t", "enter".
    """
    parts = [k.strip() for k in keys.split("+")]
    await _run_sync(pyautogui.hotkey, *parts)
    logger.debug("key_press(%r)", keys)
    return f"Pressed {keys}"


async def open_path(path: str) -> str:
    """Open a file or folder using the OS default handler.

    Args:
        path: Absolute path to the file or folder to open.
    """
    system = platform.system()
    if system == "Darwin":
        cmd = ["open", path]
    elif system == "Linux":
        cmd = ["xdg-open", path]
    elif system == "Windows":
        cmd = ["explorer", path]
    else:
        return f"Unsupported platform: {system}"

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        err = stderr.decode().strip()
        logger.error("open_path(%r) failed: %s", path, err)
        return f"Failed to open {path}: {err}"

    logger.debug("open_path(%r)", path)
    return f"Opened {path}"
