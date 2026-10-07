from enum import StrEnum


class ConnectionValidationStatus(StrEnum):
    CONNECTED = "connected"
    FAILED = "failed"
    NEEDS_REAUTHORIZATION = "needs_reauthorization"
    NEEDS_SCOPES = "needs_scopes"
    PENDING = "pending"

    def __str__(self) -> str:
        return str(self.value)
