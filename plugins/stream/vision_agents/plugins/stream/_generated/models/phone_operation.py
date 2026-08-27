from enum import Enum


class PhoneOperation(str, Enum):
    PHONE_OPERATION_ATTACH = "attach"
    PHONE_OPERATION_BUY = "buy"
    PHONE_OPERATION_DIAL = "dial"
    PHONE_OPERATION_RELEASE = "release"
    PHONE_OPERATION_SEARCH = "search"
    PHONE_OPERATION_SEND_DIGITS = "send_digits"

    def __str__(self) -> str:
        return str(self.value)
