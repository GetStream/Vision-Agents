from enum import StrEnum


class TextContentPartType(StrEnum):
    TEXT = "text"

    def __str__(self) -> str:
        return str(self.value)
