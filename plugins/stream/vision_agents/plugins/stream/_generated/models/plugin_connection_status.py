from enum import StrEnum


class PluginConnectionStatus(StrEnum):
    CONNECTED = "connected"
    FAILED = "failed"
    PENDING = "pending"

    def __str__(self) -> str:
        return str(self.value)
