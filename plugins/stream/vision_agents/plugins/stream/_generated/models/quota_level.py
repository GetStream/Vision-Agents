from enum import StrEnum


class QuotaLevel(StrEnum):
    QUOTA_LEVEL_APP = "app"
    QUOTA_LEVEL_DEFAULT = "default"
    QUOTA_LEVEL_ORGANIZATION = "organization"
    QUOTA_LEVEL_USER = "user"

    def __str__(self) -> str:
        return str(self.value)
