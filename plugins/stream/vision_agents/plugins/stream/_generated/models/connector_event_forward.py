from enum import StrEnum


class ConnectorEventForward(StrEnum):
    ALL = "all"
    UNHANDLED = "unhandled"

    def __str__(self) -> str:
        return str(self.value)
