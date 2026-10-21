from enum import StrEnum


class ScheduledTaskRunState(StrEnum):
    DELIVERED = "delivered"
    FAILED = "failed"
    MISSED = "missed"
    RUNNING = "running"

    def __str__(self) -> str:
        return str(self.value)
