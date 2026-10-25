from enum import StrEnum


class PriceChangeKind(StrEnum):
    API = "api"
    RELEASE = "release"
    SEED = "seed"

    def __str__(self) -> str:
        return str(self.value)
