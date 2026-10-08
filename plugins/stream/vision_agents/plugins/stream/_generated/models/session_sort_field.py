from enum import StrEnum


class SessionSortField(StrEnum):
    RELEVANCE = "relevance"
    UPDATED_AT = "updated_at"

    def __str__(self) -> str:
        return str(self.value)
