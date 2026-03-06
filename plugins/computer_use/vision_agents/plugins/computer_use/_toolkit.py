"""ComputerUseToolkit — registers desktop control tools on any LLM."""

import logging

from vision_agents.core.llm.llm import LLM

from . import _actions

logger = logging.getLogger(__name__)

_TOOL_DESCRIPTIONS = {
    "click": "Click at screen coordinates (x, y) with the specified mouse button.",
    "double_click": "Double-click at screen coordinates (x, y).",
    "type_text": "Type a string of text into the currently focused element.",
    "key_press": 'Press a key or key combination, e.g. "cmd+c", "enter", "ctrl+shift+t".',
    "scroll": "Scroll at screen coordinates (x, y) in the given direction.",
    "mouse_move": "Move the mouse cursor to screen coordinates (x, y).",
    "open_path": "Open a file or folder by its absolute path using the OS default handler.",
}


class ComputerUseToolkit:
    """Bundles desktop-control tools and registers them on an LLM.

    Usage::

        from vision_agents.plugins import computer_use

        computer_use.ComputerUseToolkit().register(llm)
    """

    def register(self, llm: LLM) -> None:
        """Register all computer-use action tools on *llm*."""
        tools = {
            "click": _actions.click,
            "double_click": _actions.double_click,
            "type_text": _actions.type_text,
            "key_press": _actions.key_press,
            "scroll": _actions.scroll,
            "mouse_move": _actions.mouse_move,
            "open_path": _actions.open_path,
        }
        for name, func in tools.items():
            llm.function_registry.register(
                name=name, description=_TOOL_DESCRIPTIONS[name]
            )(func)
        logger.info(
            "Registered %d computer-use tools on %s", len(tools), type(llm).__name__
        )
