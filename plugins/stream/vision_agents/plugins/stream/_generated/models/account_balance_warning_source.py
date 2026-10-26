from enum import StrEnum


class AccountBalanceWarningSource(StrEnum):
    ACCOUNT = "account"
    ORGANIZATION_POLICY = "organization_policy"
    TIER = "tier"

    def __str__(self) -> str:
        return str(self.value)
