from enum import StrEnum


class CustomModelRequestTrainsOnData(StrEnum):
    NO = "no"
    UNKNOWN = "unknown"
    YES = "yes"

    def __str__(self) -> str:
        return str(self.value)
