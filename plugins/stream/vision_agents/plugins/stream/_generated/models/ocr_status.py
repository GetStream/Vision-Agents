from enum import StrEnum


class OcrStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"

    def __str__(self) -> str:
        return str(self.value)
