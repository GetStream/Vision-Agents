from enum import StrEnum


class Harness(StrEnum):
    DEFAULT = "default"

    def __str__(self) -> str:
        return str(self.value)
