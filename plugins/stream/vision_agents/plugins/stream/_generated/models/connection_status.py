from enum import StrEnum


class ConnectionStatus(StrEnum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    NEEDS_REAUTHORIZATION = "needs_reauthorization"
    PENDING = "pending"

    def __str__(self) -> str:
        return str(self.value)
