from enum import StrEnum


class Granularity(StrEnum):
    DAILY = "daily"
    HOURLY = "hourly"

    def __str__(self) -> str:
        return str(self.value)
