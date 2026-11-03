from enum import StrEnum


class VoicemailAction(StrEnum):
    HANG_UP = "hang_up"
    LEAVE_MESSAGE = "leave_message"

    def __str__(self) -> str:
        return str(self.value)
