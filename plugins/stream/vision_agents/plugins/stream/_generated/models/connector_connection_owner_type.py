from enum import StrEnum


class ConnectorConnectionOwnerType(StrEnum):
    APP = "app"
    USER = "user"

    def __str__(self) -> str:
        return str(self.value)
