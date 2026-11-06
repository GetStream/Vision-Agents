from enum import StrEnum


class ConnectorAuditAction(StrEnum):
    GRANT_CREATED = "grant_created"
    GRANT_REFRESHED = "grant_refreshed"
    GRANT_REVOKED = "grant_revoked"
    PROXY_CALL = "proxy_call"
    TOKEN_EXPORT = "token_export"
    TOOL_REAPPROVED = "tool_reapproved"

    def __str__(self) -> str:
        return str(self.value)

    @classmethod
    def _decode(cls, value: object) -> "ConnectorAuditAction":
        """Reads the value from an answer. A string this client does not list yet is kept as a member of its own,
        so an answer from a newer router still reads, and is written back as it came."""
        try:
            return cls(value)
        except ValueError:
            if not isinstance(value, str):
                raise
            member = str.__new__(cls, value)
            member._name_ = value
            member._value_ = value
            return member
