from enum import StrEnum


class Endpointing(StrEnum):
    SEMANTIC = "semantic"
    SILENCE = "silence"

    def __str__(self) -> str:
        return str(self.value)
