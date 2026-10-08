from enum import StrEnum


class ImageErrorCode(StrEnum):
    CANCELLED = "cancelled"
    CONTENT_FILTERED = "content_filtered"
    PROVIDER_FAILED = "provider_failed"
    TIMEOUT = "timeout"
    UNSUPPORTED_OPTION = "unsupported_option"

    def __str__(self) -> str:
        return str(self.value)
