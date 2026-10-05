from enum import StrEnum


class ConnectorClientAuthMethod(StrEnum):
    CLIENT_SECRET_BASIC = "client_secret_basic"
    CLIENT_SECRET_POST = "client_secret_post"
    NONE = "none"
    PRIVATE_KEY_JWT = "private_key_jwt"
    TLS_CLIENT_AUTH = "tls_client_auth"

    def __str__(self) -> str:
        return str(self.value)
