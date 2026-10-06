from enum import StrEnum


class StreamTypeState(StrEnum):
    MISSING = "missing"
    PRESENT = "present"
    UNKNOWN = "unknown"
    UNSAFE = "unsafe"

    def __str__(self) -> str:
        return str(self.value)
