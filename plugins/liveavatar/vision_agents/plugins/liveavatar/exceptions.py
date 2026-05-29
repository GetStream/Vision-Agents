class LiveAvatarError(Exception):
    """Base exception for LiveAvatar API errors."""


class LiveAvatarAPIError(LiveAvatarError):
    """Raised when an HTTP request to the LiveAvatar API fails."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        body: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(message)
