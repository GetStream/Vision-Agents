from enum import StrEnum


class ConnectorConnectionAuthType(StrEnum):
    API_KEY = "api_key"
    BEARER = "bearer"
    NONE = "none"
    OAUTH2 = "oauth2"

    def __str__(self) -> str:
        return str(self.value)
