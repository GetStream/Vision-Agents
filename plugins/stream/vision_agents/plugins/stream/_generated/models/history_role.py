from enum import StrEnum


class HistoryRole(StrEnum):
    HISTORY_ROLE_ASSISTANT = "assistant"
    HISTORY_ROLE_USER = "user"

    def __str__(self) -> str:
        return str(self.value)
