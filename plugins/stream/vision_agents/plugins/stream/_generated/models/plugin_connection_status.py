from enum import StrEnum


class PluginConnectionStatus(StrEnum):
    CONNECTED = "connected"
    FAILED = "failed"
    NOT_CONNECTED = "not_connected"
    PENDING = "pending"

    def __str__(self) -> str:
        return str(self.value)
