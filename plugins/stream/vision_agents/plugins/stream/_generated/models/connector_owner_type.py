from enum import StrEnum


class ConnectorOwnerType(StrEnum):
    APP = "app"
    USER = "user"

    def __str__(self) -> str:
        return str(self.value)
