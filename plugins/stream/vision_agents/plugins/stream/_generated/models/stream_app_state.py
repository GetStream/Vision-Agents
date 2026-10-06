from enum import StrEnum


class StreamAppState(StrEnum):
    BLOCKED = "blocked"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"

    def __str__(self) -> str:
        return str(self.value)
