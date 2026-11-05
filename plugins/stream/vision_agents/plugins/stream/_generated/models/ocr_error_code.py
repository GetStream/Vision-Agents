from enum import StrEnum


class OcrErrorCode(StrEnum):
    CANCELLED = "cancelled"
    PROVIDER_FAILED = "provider_failed"
    TIMEOUT = "timeout"

    def __str__(self) -> str:
        return str(self.value)
