from enum import StrEnum


class LimitLevel(StrEnum):
    LIMIT_LEVEL_API_KEY = "api_key"
    LIMIT_LEVEL_APP = "app"
    LIMIT_LEVEL_ORGANIZATION = "organization"
    LIMIT_LEVEL_USER = "user"

    def __str__(self) -> str:
        return str(self.value)
