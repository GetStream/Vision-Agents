from enum import StrEnum


class SearchDepth(StrEnum):
    DEEP = "deep"
    FAST = "fast"
    INSTANT = "instant"
    STANDARD = "standard"

    def __str__(self) -> str:
        return str(self.value)
