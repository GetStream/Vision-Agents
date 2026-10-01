from enum import StrEnum


class ConnectorConnectionStatus(StrEnum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    FAILED = "failed"
    NEEDS_REAUTHORIZATION = "needs_reauthorization"
    PENDING = "pending"

    def __str__(self) -> str:
        return str(self.value)
