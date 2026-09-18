from enum import StrEnum


class SearchSessionsState(StrEnum):
    CLOSED = "closed"
    RUNNING = "running"

    def __str__(self) -> str:
        return str(self.value)
