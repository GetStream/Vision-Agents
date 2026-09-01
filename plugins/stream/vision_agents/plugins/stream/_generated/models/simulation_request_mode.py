from enum import StrEnum


class SimulationRequestMode(StrEnum):
    AUDIO = "audio"
    TEXT = "text"

    def __str__(self) -> str:
        return str(self.value)
