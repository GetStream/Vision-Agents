from enum import StrEnum


class SimulationMode(StrEnum):
    AUDIO = "audio"
    TEXT = "text"

    def __str__(self) -> str:
        return str(self.value)
