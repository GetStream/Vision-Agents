from enum import StrEnum


class ActivityGranularity(StrEnum):
    DAILY = "daily"
    MONTHLY = "monthly"

    def __str__(self) -> str:
        return str(self.value)
