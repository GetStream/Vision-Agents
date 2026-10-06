from enum import StrEnum


class AuthorizationKind(StrEnum):
    CONSENT = "consent"
    RECONNECT = "reconnect"

    def __str__(self) -> str:
        return str(self.value)
