from enum import StrEnum


class VoiceBindingState(StrEnum):
    FAILED = "failed"
    PENDING = "pending"
    READY = "ready"

    def __str__(self) -> str:
        return str(self.value)
