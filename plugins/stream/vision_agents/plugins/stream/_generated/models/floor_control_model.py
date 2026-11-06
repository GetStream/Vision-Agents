from enum import StrEnum


class FloorControlModel(StrEnum):
    CHORUS_ASR = "chorus-asr"

    def __str__(self) -> str:
        return str(self.value)
