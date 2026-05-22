from abc import ABC, abstractmethod


class Lifecycle(ABC):
    """Common lifecycle protocol for agent components.

    Anything the agent owns that needs to set up resources (network
    connections, background tasks, file handles) on join and release them
    on close inherits from this so the agent can drive both phases
    uniformly.
    """

    async def start(self) -> None:
        """Initialise the component. No-op by default — override if needed."""

    @abstractmethod
    async def close(self) -> None:
        """Release resources held by the component."""
