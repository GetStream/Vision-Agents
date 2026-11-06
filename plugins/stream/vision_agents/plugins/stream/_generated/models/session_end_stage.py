from enum import StrEnum


class SessionEndStage(StrEnum):
    CONVERSATION = "conversation"
    RINGING = "ringing"

    def __str__(self) -> str:
        return str(self.value)
