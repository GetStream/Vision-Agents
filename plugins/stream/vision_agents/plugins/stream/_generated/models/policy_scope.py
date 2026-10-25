from enum import StrEnum


class PolicyScope(StrEnum):
    POLICY_SCOPE_APP = "app"
    POLICY_SCOPE_ORGANIZATION = "organization"
    POLICY_SCOPE_USER = "user"

    def __str__(self) -> str:
        return str(self.value)
