from enum import StrEnum


class BusinessProfileRequestOrganizationType(StrEnum):
    GOVERNMENT = "government"
    NONPROFIT = "nonprofit"
    PRIVATE = "private"
    PUBLIC = "public"

    def __str__(self) -> str:
        return str(self.value)
