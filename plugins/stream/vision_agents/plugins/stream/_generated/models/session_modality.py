from enum import StrEnum


class SessionModality(StrEnum):
    TEXT = "text"
    VIDEO = "video"
    VOICE = "voice"

    def __str__(self) -> str:
        return str(self.value)
