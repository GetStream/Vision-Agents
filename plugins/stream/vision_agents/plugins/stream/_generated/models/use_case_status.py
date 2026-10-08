from enum import StrEnum


class UseCaseStatus(StrEnum):
    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"
    DRAFT = "draft"
    REJECTED = "rejected"
    SUBMITTED = "submitted"
    VENDOR_PENDING = "vendor_pending"
    VENDOR_REJECTED = "vendor_rejected"

    def __str__(self) -> str:
        return str(self.value)
