from enum import StrEnum


class ChannelIdentity(StrEnum):
    LINK = "link"
    PHONE = "phone"

    def __str__(self) -> str:
        return str(self.value)
