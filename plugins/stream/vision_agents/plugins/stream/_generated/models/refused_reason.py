from enum import StrEnum


class RefusedReason(StrEnum):
    DUPLICATE = "duplicate"
    INVALID_NUMBER = "invalid_number"
    INVALID_TIME_ZONE = "invalid_time_zone"
    INVALID_VARIABLE = "invalid_variable"

    def __str__(self) -> str:
        return str(self.value)
