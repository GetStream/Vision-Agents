from enum import StrEnum


class ConnectorAuditAction(StrEnum):
    GRANT_CREATED = "grant_created"
    GRANT_REFRESHED = "grant_refreshed"
    GRANT_REVOKED = "grant_revoked"

    def __str__(self) -> str:
        return str(self.value)
