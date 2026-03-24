"""Web search engines for the Monty sandbox."""

import logging
import re
from typing import Protocol
from urllib.parse import unquote

import aiohttp

logger = logging.getLogger(__name__)


class SearchResult:
    __slots__ = ("title", "link", "snippet")

    def __init__(self, title: str, link: str, snippet: str) -> None:
        self.title = title
        self.link = link
        self.snippet = snippet

    def to_dict(self) -> dict[str, str]:
        return {"title": self.title, "link": self.link, "snippet": self.snippet}


class SearchEngine(Protocol):
    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]: ...


class DuckDuckGoLite:
    """DuckDuckGo HTML lite — no API key, parses HTML."""

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        url = f"https://lite.duckduckgo.com/lite/?q={query.replace(' ', '+')}"
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, headers={"User-Agent": "Mozilla/5.0"}
            ) as response:
                html = await response.text()

        results: list[SearchResult] = []
        rows = re.findall(r"class='result-link'>(.*?)</a>", html)
        links = re.findall(r"uddg=(.*?)&amp;", html)
        snippets = re.findall(
            r"<td class='result-snippet'>\s*(.*?)\s*</td>", html, re.DOTALL
        )

        for i, title in enumerate(rows[:max_results]):
            title = re.sub(r"<.*?>", "", title).strip()
            link = unquote(links[i]) if i < len(links) else ""
            snippet = ""
            if i < len(snippets):
                snippet = re.sub(r"<.*?>", "", snippets[i]).strip()
            results.append(SearchResult(title=title, link=link, snippet=snippet))

        return results


# Snippets can be very long; truncate to keep tool responses within Gemini limits.
MAX_SNIPPET_LENGTH = 500


class SearXNG:
    """SearXNG — local metasearch engine, returns JSON."""

    def __init__(self, base_url: str = "http://localhost:8080") -> None:
        self._base_url = base_url

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        url = f"{self._base_url}/search"
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, params={"q": query, "format": "json"}
            ) as response:
                data = await response.json()

        results: list[SearchResult] = []
        for item in data.get("results", [])[:max_results]:
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    link=item.get("url", ""),
                    snippet=item.get("content", "")[:MAX_SNIPPET_LENGTH],
                )
            )

        return results
