from enum import StrEnum


class LimitGrantState(StrEnum):
    LIMIT_GRANT_STATE_EXPIRED = "expired"
    LIMIT_GRANT_STATE_LIVE = "live"
    LIMIT_GRANT_STATE_REVOKED = "revoked"
    LIMIT_GRANT_STATE_USED = "used"

    def __str__(self) -> str:
        return str(self.value)
