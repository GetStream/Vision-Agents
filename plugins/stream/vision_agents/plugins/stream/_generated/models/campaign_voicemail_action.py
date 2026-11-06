from enum import StrEnum


class CampaignVoicemailAction(StrEnum):
    HANG_UP = "hang_up"
    LEAVE_MESSAGE = "leave_message"
    OFF = "off"

    def __str__(self) -> str:
        return str(self.value)
