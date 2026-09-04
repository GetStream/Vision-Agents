from enum import StrEnum


class SearchOptionsContentsItem(StrEnum):
    HIGHLIGHTS = "highlights"
    SUMMARY = "summary"
    TEXT = "text"

    def __str__(self) -> str:
        return str(self.value)
