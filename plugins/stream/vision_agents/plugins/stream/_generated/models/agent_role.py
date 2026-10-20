from enum import StrEnum


class AgentRole(StrEnum):
    AGENT_MODERATOR = "agent_moderator"

    def __str__(self) -> str:
        return str(self.value)
