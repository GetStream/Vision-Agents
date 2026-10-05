from enum import StrEnum


class ConnectorConnectionOwnerType(StrEnum):
    CONNECTOR_CONNECTION_OWNER_TYPE_APP = "app"
    CONNECTOR_CONNECTION_OWNER_TYPE_USER = "user"

    def __str__(self) -> str:
        return str(self.value)
