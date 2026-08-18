from enum import Enum


class SessionState(str, Enum):
    ENDED = "ended"
    LIVE = "live"

    def __str__(self) -> str:
        return str(self.value)
