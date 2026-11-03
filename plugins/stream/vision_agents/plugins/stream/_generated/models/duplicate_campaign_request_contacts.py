from enum import StrEnum


class DuplicateCampaignRequestContacts(StrEnum):
    ALL = "all"
    NONE = "none"
    NOT_CONTACTED = "not_contacted"

    def __str__(self) -> str:
        return str(self.value)
