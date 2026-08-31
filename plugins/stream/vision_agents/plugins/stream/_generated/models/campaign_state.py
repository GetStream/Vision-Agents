from enum import StrEnum


class CampaignState(StrEnum):
    DRAFT = "draft"
    FINISHED = "finished"
    PAUSED = "paused"
    RUNNING = "running"

    def __str__(self) -> str:
        return str(self.value)
