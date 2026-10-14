from enum import StrEnum


class DetectionErrorCode(StrEnum):
    CANCELLED = "cancelled"
    PROVIDER_FAILED = "provider_failed"
    TIMEOUT = "timeout"
    UNSUPPORTED_OPTION = "unsupported_option"

    def __str__(self) -> str:
        return str(self.value)
