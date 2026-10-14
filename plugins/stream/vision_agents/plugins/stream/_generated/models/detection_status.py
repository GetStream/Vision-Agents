from enum import StrEnum


class DetectionStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"

    def __str__(self) -> str:
        return str(self.value)
