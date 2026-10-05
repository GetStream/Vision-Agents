from enum import StrEnum


class ConnectorClientRegistration(StrEnum):
    CIMD = "cimd"
    CUSTOMER = "customer"
    DCR = "dcr"
    OPERATOR = "operator"

    def __str__(self) -> str:
        return str(self.value)
