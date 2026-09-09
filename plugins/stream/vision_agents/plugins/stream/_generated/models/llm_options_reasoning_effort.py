from enum import StrEnum


class LlmOptionsReasoningEffort(StrEnum):
    HIGH = "high"
    LOW = "low"
    MEDIUM = "medium"
    MINIMAL = "minimal"

    def __str__(self) -> str:
        return str(self.value)
