from abc import ABC


class Component(ABC):
    """Base class for agent-owned components with optional start/close hooks.

    The agent calls ``start`` on every component when joining a call and
    ``close`` on every component during teardown. Both default to no-ops;
    a component overrides whichever hook it actually needs (open a
    WebSocket in ``start``, release it in ``close``, etc.).

    Inherits from :class:`ABC` so subclasses can declare their own
    ``@abstractmethod``s — ``Component`` itself has none.
    """

    async def start(self) -> None:
        """Initialise the component. No-op by default — override if needed."""

    async def close(self) -> None:
        """Release resources held by the component. No-op by default — override if needed."""
