from enum import StrEnum


class ToolResultCommandType(StrEnum):
    TOOL_RESULT = "tool_result"

    def __str__(self) -> str:
        return str(self.value)
