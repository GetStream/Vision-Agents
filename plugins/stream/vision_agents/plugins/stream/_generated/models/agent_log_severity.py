from enum import StrEnum


class AgentLogSeverity(StrEnum):
    ERROR = "error"
    INFO = "info"

    def __str__(self) -> str:
        return str(self.value)
