from enum import Enum


class CampaignState(str, Enum):
    DRAFT = "draft"
    FINISHED = "finished"
    PAUSED = "paused"
    RUNNING = "running"

    def __str__(self) -> str:
        return str(self.value)
