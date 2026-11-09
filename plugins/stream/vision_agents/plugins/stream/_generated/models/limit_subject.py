from enum import StrEnum


class LimitSubject(StrEnum):
    LIMIT_SUBJECT_API_KEY = "api_key"
    LIMIT_SUBJECT_APP = "app"
    LIMIT_SUBJECT_APP_CATEGORY = "app_category"
    LIMIT_SUBJECT_APP_TEAM = "app_team"
    LIMIT_SUBJECT_CATEGORY = "category"
    LIMIT_SUBJECT_ORGANIZATION = "organization"
    LIMIT_SUBJECT_TEAM = "team"
    LIMIT_SUBJECT_USER = "user"

    def __str__(self) -> str:
        return str(self.value)

    @classmethod
    def _decode(cls, value: object) -> "LimitSubject":
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
