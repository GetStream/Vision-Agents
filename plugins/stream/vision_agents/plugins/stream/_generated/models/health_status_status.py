from enum import StrEnum


class HealthStatusStatus(StrEnum):
    DEGRADED = "degraded"
    OK = "ok"

    def __str__(self) -> str:
        return str(self.value)
