from enum import StrEnum


class AccountStatus(StrEnum):
    ACTIVE = "active"
    SUSPENDED = "suspended"

    def __str__(self) -> str:
        return str(self.value)
