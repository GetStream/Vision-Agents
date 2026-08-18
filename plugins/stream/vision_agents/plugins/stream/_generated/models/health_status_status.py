from enum import Enum


class HealthStatusStatus(str, Enum):
    DEGRADED = "degraded"
    OK = "ok"

    def __str__(self) -> str:
        return str(self.value)
