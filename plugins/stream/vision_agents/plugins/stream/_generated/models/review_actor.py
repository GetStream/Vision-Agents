from enum import StrEnum


class ReviewActor(StrEnum):
    APP = "app"
    STAFF = "staff"
    VENDOR = "vendor"

    def __str__(self) -> str:
        return str(self.value)
