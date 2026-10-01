from enum import StrEnum


class CreateConnectorDefinitionRequestAuthMode(StrEnum):
    API_KEY = "api_key"
    BEARER = "bearer"
    NONE = "none"
    OAUTH_DCR = "oauth_dcr"

    def __str__(self) -> str:
        return str(self.value)
