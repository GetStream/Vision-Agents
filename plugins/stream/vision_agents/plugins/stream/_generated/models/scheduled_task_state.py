from enum import StrEnum


class ScheduledTaskState(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

    def __str__(self) -> str:
        return str(self.value)
