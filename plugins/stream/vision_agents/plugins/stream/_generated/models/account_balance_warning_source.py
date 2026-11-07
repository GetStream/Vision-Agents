from enum import StrEnum


class AccountBalanceWarningSource(StrEnum):
    ACCOUNT = "account"
    ORGANIZATION_POLICY = "organization_policy"
    TIER = "tier"

    def __str__(self) -> str:
        return str(self.value)

    @classmethod
    def _decode(cls, value: object) -> "AccountBalanceWarningSource":
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

    def __reduce_ex__(self, protocol: object) -> object:
        """Pickles a member this client does not list through _decode, as Enum's own reduce would call cls(value)."""
        if self._value_ in type(self)._value2member_map_:
            return super().__reduce_ex__(protocol)
        return type(self)._decode, (self._value_,)
