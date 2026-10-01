from enum import StrEnum


class AgentConnectorSelectionType(StrEnum):
    FIXED = "fixed"
    SESSION = "session"

    def __str__(self) -> str:
        return str(self.value)
