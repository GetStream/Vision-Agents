from enum import StrEnum


class CreditEntryKind(StrEnum):
    ADJUSTMENT = "adjustment"
    DISPUTE = "dispute"
    DISPUTE_REVERSAL = "dispute_reversal"
    GRANT = "grant"
    REFUND = "refund"
    TOP_UP = "top_up"

    def __str__(self) -> str:
        return str(self.value)
