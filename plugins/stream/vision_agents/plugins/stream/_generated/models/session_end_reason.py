from enum import StrEnum


class SessionEndReason(StrEnum):
    AGENT_FAILED = "agent_failed"
    AGENT_HANGUP = "agent_hangup"
    CLOSED = "closed"
    HANGUP = "hangup"
    MODEL_CONFIG_REFUSED = "model_config_refused"
    MODEL_FAILED = "model_failed"
    MODEL_NO_CREDIT = "model_no_credit"
    SHUTDOWN = "shutdown"
    START = "start"
    TRANSFERRED = "transferred"
    UNANSWERED = "unanswered"
    VOICEMAIL = "voicemail"

    def __str__(self) -> str:
        return str(self.value)
