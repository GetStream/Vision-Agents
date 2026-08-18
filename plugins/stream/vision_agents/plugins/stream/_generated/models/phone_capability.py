from enum import Enum


class PhoneCapability(str, Enum):
    FAX = "fax"
    MMS = "mms"
    SMS = "sms"
    VOICE = "voice"

    def __str__(self) -> str:
        return str(self.value)
