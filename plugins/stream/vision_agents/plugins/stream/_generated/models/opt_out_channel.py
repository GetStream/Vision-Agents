from enum import StrEnum


class OptOutChannel(StrEnum):
    ALL = "all"
    IMESSAGE = "imessage"
    SMS = "sms"
    VOICE = "voice"
    WHATSAPP = "whatsapp"

    def __str__(self) -> str:
        return str(self.value)
