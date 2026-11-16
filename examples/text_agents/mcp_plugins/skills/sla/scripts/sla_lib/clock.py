"""Adding minutes to a time on a clock that may pause outside business hours."""

from datetime import datetime, timedelta


def parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def due(opened: datetime, minutes: int, clock: str, hours: dict) -> datetime:
    """When `minutes` of the clock have run, starting at `opened`."""
    if clock == "always":
        return opened + timedelta(minutes=minutes)

    left = timedelta(minutes=minutes)
    at = opened
    while True:
        start = at.replace(hour=hours["start"], minute=0, second=0, microsecond=0)
        end = at.replace(hour=hours["end"], minute=0, second=0, microsecond=0)
        if at.weekday() not in hours["weekdays"] or at >= end:
            at = start + timedelta(days=1)
            continue
        at = max(at, start)
        if at + left <= end:
            return at + left
        left -= end - at
        at = start + timedelta(days=1)
