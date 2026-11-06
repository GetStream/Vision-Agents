from enum import StrEnum


class PhoneOperation(StrEnum):
    PHONE_OPERATION_ATTACH = "attach"
    PHONE_OPERATION_BUY = "buy"
    PHONE_OPERATION_DIAL = "dial"
    PHONE_OPERATION_RELEASE = "release"
    PHONE_OPERATION_SEARCH = "search"
    PHONE_OPERATION_SEND_DIGITS = "send_digits"

    def __str__(self) -> str:
        return str(self.value)

    @classmethod
    def _decode(cls, value: object) -> "PhoneOperation":
        """Reads the value from an answer. A string this client does not list yet is kept as a member of its own,
        so an answer from a newer router still reads, and is written back as it came."""
        try:
            return cls(value)
        except ValueError:
            if not isinstance(value, str):
                raise
            member = str.__new__(cls, value)
            member._name_ = value
            member._value_ = value
            return member
