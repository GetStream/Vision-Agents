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

    @classmethod
    def _decode(cls, value: object) -> "CallOutcome":
        """Reads the value from an answer. A string this client does not list yet is kept as a member of its own,
        so an answer from a newer router still reads, and is written back as it came."""
        try:
            return cls(value)
        except ValueError:
            if not isinstance(value, str):
                raise
            member = str.__new__(cls, value)
            member._name_ = value
            member._value_ = value
            return member
