from enum import StrEnum


class AccountBalanceState(StrEnum):
    EXHAUSTED = "exhausted"
    LOW = "low"
    OK = "ok"

    def __str__(self) -> str:
        return str(self.value)
