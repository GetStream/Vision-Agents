from enum import StrEnum


class SessionMode(StrEnum):
    CASCADE = "cascade"
    NATIVE = "native"
    TEXT = "text"

    def __str__(self) -> str:
        return str(self.value)
