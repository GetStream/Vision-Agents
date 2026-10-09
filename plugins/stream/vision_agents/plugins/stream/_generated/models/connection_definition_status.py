from enum import StrEnum


class ConnectionDefinitionStatus(StrEnum):
    BROKEN = "broken"
    CURRENT = "current"
    OUTDATED = "outdated"

    def __str__(self) -> str:
        return str(self.value)
