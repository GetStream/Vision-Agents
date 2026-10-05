from enum import StrEnum


class UpdateSessionRequestThinking(StrEnum):
    HIGH = "high"
    LOW = "low"
    MEDIUM = "medium"
    MINIMAL = "minimal"
    NONE = "none"

    def __str__(self) -> str:
        return str(self.value)
