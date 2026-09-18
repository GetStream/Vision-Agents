from enum import StrEnum


class ListSessionsState(StrEnum):
    CLOSED = "closed"
    RUNNING = "running"

    def __str__(self) -> str:
        return str(self.value)
