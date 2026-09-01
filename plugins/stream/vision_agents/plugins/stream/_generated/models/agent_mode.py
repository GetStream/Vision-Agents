from enum import StrEnum


class AgentMode(StrEnum):
    TEXT = "text"
    VOICE = "voice"

    def __str__(self) -> str:
        return str(self.value)
