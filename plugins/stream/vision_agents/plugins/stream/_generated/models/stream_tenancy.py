from enum import StrEnum


class StreamTenancy(StrEnum):
    APP = "app"
    DEPLOYMENT = "deployment"

    def __str__(self) -> str:
        return str(self.value)
