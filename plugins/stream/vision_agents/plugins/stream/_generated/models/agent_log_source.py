from enum import StrEnum


class AgentLogSource(StrEnum):
    AGENT = "agent"
    SYSTEM = "system"
    TOOL = "tool"
    USER = "user"

    def __str__(self) -> str:
        return str(self.value)
