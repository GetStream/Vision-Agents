from enum import StrEnum


class BusinessProfileLegalEntityType(StrEnum):
    CORPORATION = "corporation"
    LLC = "llc"
    OTHER = "other"
    PARTNERSHIP = "partnership"
    SOLE_PROPRIETOR = "sole_proprietor"

    def __str__(self) -> str:
        return str(self.value)
