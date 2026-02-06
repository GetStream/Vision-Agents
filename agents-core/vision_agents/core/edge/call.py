from typing import Protocol


class Call(Protocol):
    """Protocol for call/room abstraction.

    Any EdgeTransport implementation must return objects conforming to this protocol
    from their create_call or join methods.
    """

    @property
    def id(self) -> str:
        """The unique identifier of the call."""
        ...
