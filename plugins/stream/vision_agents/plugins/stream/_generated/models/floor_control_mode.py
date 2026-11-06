from enum import StrEnum


class FloorControlMode(StrEnum):
    ACT = "act"
    OFF = "off"
    SHADOW = "shadow"

    def __str__(self) -> str:
        return str(self.value)
