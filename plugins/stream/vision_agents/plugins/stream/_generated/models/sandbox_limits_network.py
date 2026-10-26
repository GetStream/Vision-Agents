from enum import StrEnum


class SandboxLimitsNetwork(StrEnum):
    ALLOW_LIST = "allow_list"
    ANY = "any"
    NONE = "none"

    def __str__(self) -> str:
        return str(self.value)
