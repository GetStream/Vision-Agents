from enum import StrEnum


class SessionEndReason(StrEnum):
    AGENT_HANGUP = "agent_hangup"
    HANGUP = "hangup"
    START = "start"
    TRANSFERRED = "transferred"
    UNANSWERED = "unanswered"
    VOICEMAIL = "voicemail"

    def __str__(self) -> str:
        return str(self.value)
