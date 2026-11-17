class SessionLimitExceeded(Exception):
    pass


class MaxConcurrentSessionsExceeded(SessionLimitExceeded): ...


class MaxSessionsPerCallExceeded(SessionLimitExceeded): ...
