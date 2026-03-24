"""Monty sandbox — external functions and type stubs."""

from pathlib import Path
from typing import Any

from .search import DuckDuckGoLite, SearXNG, SearchEngine
from .tools import (
    fetch,
    json_dumps,
    json_parse,
    set_search_engine,
    web_search,
)

EXTERNAL_FUNCTIONS: dict[str, Any] = {
    "fetch": fetch,
    "json_parse": json_parse,
    "json_dumps": json_dumps,
    "web_search": web_search,
}

TYPE_STUBS = (Path(__file__).parent / "stubs.py").read_text()

__all__ = [
    "EXTERNAL_FUNCTIONS",
    "TYPE_STUBS",
    "DuckDuckGoLite",
    "SearXNG",
    "SearchEngine",
    "set_search_engine",
]
