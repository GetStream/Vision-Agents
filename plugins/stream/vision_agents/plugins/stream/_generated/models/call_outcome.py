from enum import StrEnum


class CallOutcome(StrEnum):
    BUSY = "busy"
    CANCELED = "canceled"
    COMPLETED = "completed"
    FAILED = "failed"
    NO_ANSWER = "no_answer"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return str(self.value)
