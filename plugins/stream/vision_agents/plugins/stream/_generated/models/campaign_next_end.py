from enum import StrEnum


class CampaignNextEnd(StrEnum):
    SUCCESSFUL = "successful"
    UNSUCCESSFUL = "unsuccessful"

    def __str__(self) -> str:
        return str(self.value)
