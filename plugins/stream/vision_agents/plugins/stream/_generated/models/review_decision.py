from enum import StrEnum


class ReviewDecision(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_CHANGES = "request_changes"

    def __str__(self) -> str:
        return str(self.value)
