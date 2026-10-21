from enum import StrEnum


class ScheduledTaskCreatedBy(StrEnum):
    SCHEDULED_TASK_CREATED_BY_AGENT = "agent"
    SCHEDULED_TASK_CREATED_BY_API = "api"

    def __str__(self) -> str:
        return str(self.value)
