"""Reading provider JSON without trusting its shape."""

import re
from datetime import datetime, timezone
from typing import Optional

_FRACTION = re.compile(r"\.(\d+)")


def obj(value: object) -> dict[str, object]:
    """The value if it is a JSON object, else an empty one."""
    return value if isinstance(value, dict) else {}


def objs(value: object) -> list[dict[str, object]]:
    """The JSON objects in the value if it is an array, else none."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def string(value: object) -> str:
    """The value if it is a string, else an empty one."""
    return value if isinstance(value, str) else ""


def integer(value: object) -> Optional[int]:
    """The value as an int when it is one or a string of digits."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def number(value: object) -> Optional[float]:
    """The value as a float when it is a JSON number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def iso_time(value: object) -> Optional[datetime]:
    """An RFC 3339 time, which providers write with up to nanoseconds and a `Z`."""
    text = string(value).replace("Z", "+00:00")
    # datetime before Python 3.11 reads at most six fractional digits.
    text = _FRACTION.sub(lambda match: "." + match.group(1)[:6].ljust(6, "0"), text)
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def unix_time(value: object) -> Optional[datetime]:
    """A time written as seconds since the epoch, as a number or a string."""
    seconds = number(value)
    if seconds is None:
        try:
            seconds = float(string(value))
        except ValueError:
            return None
    return datetime.fromtimestamp(seconds, timezone.utc)


def with_links(text: str, links: list[str]) -> str:
    """The text with each link on a line of its own after it."""
    return "\n".join(part for part in [text, *links] if part)
