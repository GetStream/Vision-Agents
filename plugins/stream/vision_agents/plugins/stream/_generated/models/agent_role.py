from enum import StrEnum


class AgentRole(StrEnum):
    AI_MODERATOR = "ai_moderator"

    def __str__(self) -> str:
        return str(self.value)
