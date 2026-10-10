from enum import StrEnum


class GreetingMode(StrEnum):
    EXACT = "exact"
    VARIATION = "variation"

    def __str__(self) -> str:
        return str(self.value)
