from enum import StrEnum


class StreamKeyStateStatus(StrEnum):
    ACTIVE = "active"
    REJECTED = "rejected"

    def __str__(self) -> str:
        return str(self.value)
