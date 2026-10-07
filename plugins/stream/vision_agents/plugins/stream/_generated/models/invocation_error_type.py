from enum import StrEnum


class InvocationErrorType(StrEnum):
    CLIENT_TIMEOUT = "client_timeout"
    CUSTOMER_AUTH = "customer_auth"
    DENIED = "denied"
    EXTERNAL_SERVER = "external_server"
    OUTCOME_UNKNOWN = "outcome_unknown"

    def __str__(self) -> str:
        return str(self.value)
