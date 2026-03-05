class LemonSliceError(Exception):
    """Base exception for LemonSlice API errors."""


class LemonSliceSessionError(LemonSliceError):
    """Raised when session creation or management fails."""

    def __init__(self, message: str, status_code: int | None = None):
        self.status_code = status_code
        super().__init__(message)
