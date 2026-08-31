from enum import StrEnum


class PhoneCapability(StrEnum):
    EMERGENCY = "emergency"
    FAX = "fax"
    HD_VOICE = "hd_voice"
    INTERNATIONAL_SMS = "international_sms"
    LOCAL_CALLING = "local_calling"
    MMS = "mms"
    SMS = "sms"
    VOICE = "voice"

    def __str__(self) -> str:
        return str(self.value)
