from enum import StrEnum


class ToolApprovalCommandType(StrEnum):
    TOOL_APPROVAL = "tool_approval"

    def __str__(self) -> str:
        return str(self.value)
