from enum import StrEnum


class ImageGenerationStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"

    def __str__(self) -> str:
        return str(self.value)
