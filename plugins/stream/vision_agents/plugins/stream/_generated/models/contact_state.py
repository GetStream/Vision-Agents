from enum import Enum


class ContactState(str, Enum):
    CALLING = "calling"
    DONE = "done"
    FAILED = "failed"
    PENDING = "pending"

    def __str__(self) -> str:
        return str(self.value)
