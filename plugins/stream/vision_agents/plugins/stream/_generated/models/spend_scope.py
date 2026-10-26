from enum import StrEnum


class SpendScope(StrEnum):
    APP = "app"
    ORGANIZATION = "organization"

    def __str__(self) -> str:
        return str(self.value)
