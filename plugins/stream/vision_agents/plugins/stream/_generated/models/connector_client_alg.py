from enum import StrEnum


class ConnectorClientAlg(StrEnum):
    PS256 = "PS256"
    RS256 = "RS256"

    def __str__(self) -> str:
        return str(self.value)
