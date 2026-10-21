from enum import StrEnum


class TranscriptMessagePurpose(StrEnum):
    CHECKIN = "checkin"
    GREETING = "greeting"
    UPDATE = "update"

    def __str__(self) -> str:
        return str(self.value)
