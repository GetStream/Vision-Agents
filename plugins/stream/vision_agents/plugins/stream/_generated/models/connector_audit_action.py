from enum import StrEnum


class ConnectorAuditAction(StrEnum):
    GRANT_CREATED = "grant_created"
    GRANT_REFRESHED = "grant_refreshed"
    GRANT_REVOKED = "grant_revoked"
    PROXY_CALL = "proxy_call"
    TOKEN_EXPORT = "token_export"

    def __str__(self) -> str:
        return str(self.value)
