from enum import IntEnum


class SessionSortDirection(IntEnum):
    VALUE_NEGATIVE_1 = -1

    def __str__(self) -> str:
        return str(self.value)
