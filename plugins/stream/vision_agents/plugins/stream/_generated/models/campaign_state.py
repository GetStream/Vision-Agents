from enum import StrEnum


class CampaignState(StrEnum):
    COMPLETED = "completed"
    DRAFT = "draft"
    PAUSED = "paused"
    RUNNING = "running"
    SCHEDULED = "scheduled"
    STOPPED = "stopped"

    def __str__(self) -> str:
        return str(self.value)
