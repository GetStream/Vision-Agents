from enum import StrEnum


class ContactState(StrEnum):
    CALLING = "calling"
    DONE = "done"
    FAILED = "failed"
    PENDING = "pending"

    def __str__(self) -> str:
        return str(self.value)
