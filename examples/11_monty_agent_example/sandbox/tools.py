"""External functions exposed to the Monty sandbox."""

import json
import logging
from typing import Any

import aiohttp

from .search import SearchEngine

logger = logging.getLogger(__name__)

# Large HTTP responses can blow Gemini's connection memory limit (1GB).
MAX_FETCH_RESPONSE_SIZE = 50_000

_search_engine: SearchEngine | None = None


def set_search_engine(engine: SearchEngine) -> None:
    """Set the search engine used by web_search()."""
    global _search_engine
    _search_engine = engine


async def fetch(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: str | None = None,
) -> str:
    """Make an HTTP request. Returns response body as string (max 50KB)."""
    logger.info(f"fetch: {method} {url}")
    async with aiohttp.ClientSession() as session:
        async with session.request(method, url, headers=headers, data=body) as response:
            text = await response.text()
            if len(text) > MAX_FETCH_RESPONSE_SIZE:
                text = text[:MAX_FETCH_RESPONSE_SIZE] + "\n... (truncated)"
            logger.info(f"fetch: {response.status} ({len(text)} bytes)")
            return text


def json_parse(text: str) -> dict[str, Any] | list[Any]:
    """Parse a JSON string into a Python object."""
    return json.loads(text)


def json_dumps(obj: dict[str, Any] | list[Any]) -> str:
    """Convert a Python object to a JSON string."""
    return json.dumps(obj)


async def web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search the web and return parsed results."""
    if _search_engine is None:
        raise RuntimeError("No search engine configured. Call set_search_engine() first.")
    logger.info(f"web_search: {query}")
    results = await _search_engine.search(query, max_results)
    logger.info(f"web_search: {len(results)} results")
    return [r.to_dict() for r in results]
