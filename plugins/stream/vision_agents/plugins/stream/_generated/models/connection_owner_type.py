from enum import StrEnum


class ConnectionOwnerType(StrEnum):
    CONNECTION_OWNER_TYPE_APP = "app"
    CONNECTION_OWNER_TYPE_USER = "user"

    def __str__(self) -> str:
        return str(self.value)
