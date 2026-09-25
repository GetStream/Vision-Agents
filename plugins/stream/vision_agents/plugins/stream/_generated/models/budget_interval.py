from enum import StrEnum


class BudgetInterval(StrEnum):
    DAILY = "daily"
    HOURLY = "hourly"
    MONTHLY = "monthly"
    WEEKLY = "weekly"

    def __str__(self) -> str:
        return str(self.value)
