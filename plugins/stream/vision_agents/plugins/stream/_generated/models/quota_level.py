from enum import StrEnum


class QuotaLevel(StrEnum):
    QUOTA_LEVEL_APP = "app"
    QUOTA_LEVEL_DEFAULT = "default"
    QUOTA_LEVEL_ORGANIZATION = "organization"
    QUOTA_LEVEL_USER = "user"

    def __str__(self) -> str:
        return str(self.value)

    @classmethod
    def _decode(cls, value: object) -> "QuotaLevel":
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
