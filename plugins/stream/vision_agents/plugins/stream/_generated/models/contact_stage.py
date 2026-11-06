from enum import StrEnum


class ContactStage(StrEnum):
    COMPLETED = "completed"
    ENROLLED = "enrolled"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"
    REJECTED = "rejected"
    STOPPED = "stopped"

    def __str__(self) -> str:
        return str(self.value)
