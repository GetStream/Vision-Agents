from enum import StrEnum


class AuditAction(StrEnum):
    CREATED = "created"
    DELETED = "deleted"
    SYNCED = "synced"
    UPDATED = "updated"

    def __str__(self) -> str:
        return str(self.value)
