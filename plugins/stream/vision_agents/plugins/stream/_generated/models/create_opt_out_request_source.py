from enum import StrEnum


class CreateOptOutRequestSource(StrEnum):
    API = "api"
    DASHBOARD = "dashboard"

    def __str__(self) -> str:
        return str(self.value)
