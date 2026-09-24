"""Print Artificial Analysis TTS and streaming STT numbers as tab-separated rows.

Run with `uv run --no-project .claude/skills/refresh_model_stats/aa_stats.py`.
"""

import json
import re
import urllib.request
from collections.abc import Iterator

BASE = "https://artificialanalysis.ai/"
PUSH = re.compile(r'self\.__next_f\.push\(\[1,(".*?")\]\)</script>', re.S)
Json = dict[str, "Json"] | list["Json"] | str | float | int | bool | None


def flight(path: str) -> Iterator[Json]:
    """Yields every JSON value in the page's React Server Components payload."""
    request = urllib.request.Request(BASE + path, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response:
        html = response.read().decode()
    payload = "".join(json.loads(chunk) for chunk in PUSH.findall(html))
    for line in payload.splitlines():
        _, _, body = line.partition(":")
        try:
            yield json.loads(body)
        except json.JSONDecodeError:
            continue


def dicts(value: Json) -> Iterator[dict[str, Json]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from dicts(child)


def tts() -> None:
    print("# tts: model, host, elo, characters per second, representative host")
    seen = set()
    for value in flight("text-to-speech"):
        for d in dicts(value):
            model, host, performance = (
                d.get("model"),
                d.get("host"),
                d.get("performance"),
            )
            if not (
                isinstance(model, dict)
                and isinstance(host, dict)
                and isinstance(performance, dict)
                and "qualityElo" in model
            ):
                continue
            key = (model["name"], host["name"])
            if key in seen:
                continue
            seen.add(key)
            elo = model["qualityElo"]
            cps = performance.get("medianCharactersPerSecond")
            print(
                model["name"],
                host["name"],
                round(elo) if isinstance(elo, float) else "",
                round(cps) if isinstance(cps, float) else "",
                d.get("isRepresentative", ""),
                sep="\t",
            )


def stt() -> None:
    print("# stt: model, host, streaming AA-WER %, time to final transcript ms")
    seen = set()
    for value in flight("speech-to-text/streaming"):
        for d in dicts(value):
            wer, final = (
                d.get("aaWerStreamingIndex"),
                d.get("timeToFinalTranscriptSeconds"),
            )
            host = d.get("host")
            if not isinstance(wer, float) or not isinstance(host, dict):
                continue
            key = (d["name"], host["name"])
            if key in seen:
                continue
            seen.add(key)
            print(
                d["name"],
                host["name"],
                round(wer * 100, 1),
                round(final * 1000) if isinstance(final, float) else "",
                sep="\t",
            )


def search() -> None:
    print("# search: provider, variant, search index, cost per task USD")
    seen = set()
    for value in flight("agents/search-api"):
        for d in dicts(value):
            index = d.get("aaSearchQualityIndex")
            if not isinstance(index, float) or d.get("isBaseline"):
                continue
            key = (d["providerName"], d["providerVariant"])
            if key in seen:
                continue
            seen.add(key)
            search_cost = d.get("searchApiCostPerTaskUsd")
            model_cost = d.get("candidateModelCostPerTaskUsd")
            cost = (
                search_cost + model_cost
                if isinstance(search_cost, float) and isinstance(model_cost, float)
                else ""
            )
            print(
                d["providerName"],
                d["providerVariant"],
                round(index * 100),
                round(cost, 3) if cost != "" else "",
                sep="\t",
            )


if __name__ == "__main__":
    tts()
    stt()
    search()
