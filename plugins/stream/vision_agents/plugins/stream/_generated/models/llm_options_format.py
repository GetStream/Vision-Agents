from enum import StrEnum


class LlmOptionsFormat(StrEnum):
    JSON_OBJECT = "json_object"
    TEXT = "text"

    def __str__(self) -> str:
        return str(self.value)
