from enum import StrEnum


class ConnectorValidationStatus(StrEnum):
    CONNECTED = "connected"
    FAILED = "failed"
    NEEDS_REAUTHORIZATION = "needs_reauthorization"

    def __str__(self) -> str:
        return str(self.value)
