from enum import StrEnum


class PhoneNumberType(StrEnum):
    LOCAL = "local"
    MOBILE = "mobile"
    TOLL_FREE = "toll_free"

    def __str__(self) -> str:
        return str(self.value)
