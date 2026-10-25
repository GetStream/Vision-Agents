from enum import StrEnum


class LimitSubject(StrEnum):
    LIMIT_SUBJECT_API_KEY = "api_key"
    LIMIT_SUBJECT_APP = "app"
    LIMIT_SUBJECT_APP_CATEGORY = "app_category"
    LIMIT_SUBJECT_APP_TEAM = "app_team"
    LIMIT_SUBJECT_CATEGORY = "category"
    LIMIT_SUBJECT_ORGANIZATION = "organization"
    LIMIT_SUBJECT_TEAM = "team"
    LIMIT_SUBJECT_USER = "user"

    def __str__(self) -> str:
        return str(self.value)
