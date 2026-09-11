from enum import StrEnum


class SessionRespondCommandType(StrEnum):
    RESPOND = "respond"

    def __str__(self) -> str:
        return str(self.value)
