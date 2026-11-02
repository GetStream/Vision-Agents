from enum import StrEnum


class SessionEndReason(StrEnum):
    AGENT_HANGUP = "agent_hangup"
    HANGUP = "hangup"
    START = "start"
    TRANSFERRED = "transferred"
    UNANSWERED = "unanswered"

    def __str__(self) -> str:
        return str(self.value)
