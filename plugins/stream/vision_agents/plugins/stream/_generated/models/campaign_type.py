from enum import StrEnum


class CampaignType(StrEnum):
    AUTO_ENROLL = "auto_enroll"
    STANDARD = "standard"

    def __str__(self) -> str:
        return str(self.value)
