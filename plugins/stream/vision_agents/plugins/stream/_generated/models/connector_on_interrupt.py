from enum import StrEnum


class ConnectorOnInterrupt(StrEnum):
    CANCEL = "cancel"
    WAIT = "wait"

    def __str__(self) -> str:
        return str(self.value)
