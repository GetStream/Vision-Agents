from enum import StrEnum


class DispatchSetting(StrEnum):
    DISABLED = "disabled"
    ENABLED = "enabled"

    def __str__(self) -> str:
        return str(self.value)
