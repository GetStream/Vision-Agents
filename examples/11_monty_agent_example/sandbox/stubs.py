"""Type stubs for Monty sandbox type checking.

This file is read as a string at runtime and passed to Monty's type checker.
Write it as normal Python so the editor renders it properly.
"""

from typing import Any


async def fetch(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: str | None = None,
) -> str:
    """Make an HTTP request and return the response body as a string."""
    ...


def json_parse(text: str) -> dict[str, Any] | list[Any]:
    """Parse a JSON string into a dict or list."""
    ...


def json_dumps(obj: dict[str, Any] | list[Any]) -> str:
    """Convert a dict or list to a JSON string."""
    ...


async def web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search the web. Returns list of dicts with 'title', 'link', 'snippet'."""
    ...
