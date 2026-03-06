"""Low-level desktop actions backed by pyautogui."""

import asyncio
import logging
import platform
import subprocess
from typing import Literal

import pyautogui

logger = logging.getLogger(__name__)

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1


def _run_sync(func, *args, **kwargs):
    """Run a blocking pyautogui call in a thread executor."""
    loop = asyncio.get_running_loop()
    return loop.run_in_executor(None, lambda: func(*args, **kwargs))


async def click(
    x: int,
    y: int,
    button: str = "left",
) -> str:
    """Click at the given screen coordinates.

    Args:
        x: Horizontal pixel coordinate.
        y: Vertical pixel coordinate.
        button: Mouse button — "left", "right", or "middle".
    """
    await _run_sync(pyautogui.click, x, y, button=button)
    logger.debug("click(%d, %d, button=%s)", x, y, button)
    return f"Clicked at ({x}, {y}) with {button} button"


async def double_click(x: int, y: int) -> str:
    """Double-click at the given screen coordinates.

    Args:
        x: Horizontal pixel coordinate.
        y: Vertical pixel coordinate.
    """
    await _run_sync(pyautogui.doubleClick, x, y)
    logger.debug("double_click(%d, %d)", x, y)
    return f"Double-clicked at ({x}, {y})"


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


async def scroll(
    x: int,
    y: int,
    clicks: int = 3,
    direction: Literal["up", "down"] = "down",
) -> str:
    """Scroll at the given screen coordinates.

    Args:
        x: Horizontal pixel coordinate to scroll at.
        y: Vertical pixel coordinate to scroll at.
        clicks: Number of scroll increments.
        direction: "up" or "down".
    """
    amount = clicks if direction == "up" else -clicks
    await _run_sync(pyautogui.scroll, amount, x=x, y=y)
    logger.debug("scroll(%d, %d, clicks=%d, direction=%s)", x, y, clicks, direction)
    return f"Scrolled {direction} {clicks} clicks at ({x}, {y})"


async def mouse_move(x: int, y: int) -> str:
    """Move the mouse cursor to the given screen coordinates.

    Args:
        x: Horizontal pixel coordinate.
        y: Vertical pixel coordinate.
    """
    await _run_sync(pyautogui.moveTo, x, y)
    logger.debug("mouse_move(%d, %d)", x, y)
    return f"Moved mouse to ({x}, {y})"


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
