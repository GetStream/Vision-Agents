from enum import StrEnum


class LimitInterval(StrEnum):
    LIMIT_INTERVAL_BIWEEKLY = "biweekly"
    LIMIT_INTERVAL_DAILY = "daily"
    LIMIT_INTERVAL_MONTHLY = "monthly"
    LIMIT_INTERVAL_WEEKLY = "weekly"

    def __str__(self) -> str:
        return str(self.value)
