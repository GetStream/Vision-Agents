from enum import StrEnum


class ChannelKind(StrEnum):
    IMESSAGE = "imessage"
    SMS = "sms"
    WHATSAPP = "whatsapp"

    def __str__(self) -> str:
        return str(self.value)
