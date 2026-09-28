from enum import StrEnum


class ConnectorDefinitionAuthMode(StrEnum):
    API_KEY = "api_key"
    BEARER = "bearer"
    NONE = "none"
    OAUTH_CUSTOMER_CREDENTIALS = "oauth_customer_credentials"
    OAUTH_DCR = "oauth_dcr"
    OAUTH_PRECONFIGURED = "oauth_preconfigured"

    def __str__(self) -> str:
        return str(self.value)
