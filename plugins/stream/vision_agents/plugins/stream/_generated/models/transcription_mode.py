from enum import StrEnum


class TranscriptionMode(StrEnum):
    SMART = "smart"
    VERBATIM = "verbatim"

    def __str__(self) -> str:
        return str(self.value)
