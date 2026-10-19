from enum import StrEnum


class ScheduledTaskCreatedBy(StrEnum):
    AGENT = "agent"
    API = "api"

    def __str__(self) -> str:
        return str(self.value)
