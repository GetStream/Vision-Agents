from enum import StrEnum


class ImageSourceDetail(StrEnum):
    AUTO = "auto"
    HIGH = "high"
    LOW = "low"

    def __str__(self) -> str:
        return str(self.value)
