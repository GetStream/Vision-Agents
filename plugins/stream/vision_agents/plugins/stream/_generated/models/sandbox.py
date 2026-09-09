from enum import StrEnum


class Sandbox(StrEnum):
    DAYTONA = "daytona"

    def __str__(self) -> str:
        return str(self.value)
