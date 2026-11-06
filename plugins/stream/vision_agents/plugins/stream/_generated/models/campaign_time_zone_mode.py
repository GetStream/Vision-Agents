from enum import StrEnum


class CampaignTimeZoneMode(StrEnum):
    CONTACT = "contact"
    FIXED = "fixed"

    def __str__(self) -> str:
        return str(self.value)
