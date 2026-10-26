"""Print LiteLLM's figures for the router's price entries, in the router's meters and units.

Run with `uv run --no-project .claude/skills/check_prices/litellm.py [--list path] [--json]`.

It reads LiteLLM's model_prices_and_context_window.json at LiteLLM's latest release tag (never
main) and prints the tag. --list is a price file or the output of `router prices show --json`;
the default is acceleration/prices/vendor.json. LiteLLM is a second opinion: where it and the
vendor's own page disagree, the page wins.
"""

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

RELEASES = "https://api.github.com/repos/BerriAI/litellm/releases/latest"
CATALOG = "https://raw.githubusercontent.com/BerriAI/litellm/{tag}/model_prices_and_context_window.json"
VENDOR = Path(__file__).resolve().parents[3] / "acceleration" / "prices" / "vendor.json"

# Our provider names where LiteLLM's prefix differs.
PREFIXES = {"together": "together_ai", "grok": "xai", "fal": "fal_ai"}

# LiteLLM's per-unit fields, as (our meter, how many units our rate is per, the multiplier
# from LiteLLM's unit to ours).
FIELDS = {
    "input_cost_per_token": ("input_tokens", 1_000_000, 1_000_000),
    "cache_read_input_token_cost": ("cached_input_tokens", 1_000_000, 1_000_000),
    "cache_creation_input_token_cost": (
        "cache_write_input_tokens",
        1_000_000,
        1_000_000,
    ),
    "input_cost_per_audio_token": ("input_audio_tokens", 1_000_000, 1_000_000),
    "output_cost_per_token": ("output_tokens", 1_000_000, 1_000_000),
    "output_cost_per_audio_token": ("output_audio_tokens", 1_000_000, 1_000_000),
    "input_cost_per_second": ("audio_ms", 3_600_000, 3_600),
    "input_cost_per_character": ("characters", 1_000_000, 1_000_000),
    "output_cost_per_character": ("characters (output)", 1_000_000, 1_000_000),
    "output_cost_per_image": ("images", 1, 1),
}
ABOVE = re.compile(r"^(.+)_above_(\d+)k_tokens$")

# The suffix LiteLLM gives a field priced at a service tier, by the tier our entries name in
# `when.service_tier`. OpenAI bills a `fast` request as `priority`.
TIER_SUFFIXES = {
    "flex": "_flex",
    "batch": "_batches",
    "priority": "_priority",
    "fast": "_priority",
    "ultrafast": "_ultrafast",
}
SUFFIXED = re.compile(r"_(flex|batches|priority|ultrafast|decisions)$")

Json = dict[str, "Json"] | list["Json"] | str | float | int | bool | None


def fetch(url: str) -> Json:
    """Reads a JSON document over HTTPS."""
    request = urllib.request.Request(url, headers={"User-Agent": "check_prices"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode())


def latest_tag() -> str:
    """Returns LiteLLM's latest release tag."""
    release = fetch(RELEASES)
    if not isinstance(release, dict) or not isinstance(release.get("tag_name"), str):
        raise SystemExit("no release tag in " + RELEASES)
    return release["tag_name"]


def match(catalog: dict[str, Json], provider: str, model: str) -> str | None:
    """Returns LiteLLM's key for an item: <provider>/<model>, then <model>."""
    prefix = PREFIXES.get(provider, provider)
    for key in (f"{prefix}/{model}", f"{provider}/{model}", model):
        if isinstance(catalog.get(key), dict):
            return key
    return None


def figures(entry: dict[str, Json], tier: str) -> dict[str, tuple[float, int]]:
    """Returns LiteLLM's figures for one catalog entry at a service tier ("" for the default),
    by our meter (or context tier), as (usd, per)."""
    found: dict[str, tuple[float, int]] = {}
    suffix = TIER_SUFFIXES.get(tier, "")
    for field, value in entry.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if suffix:
            if not field.endswith(suffix):
                continue
            field = field.removesuffix(suffix)
        elif SUFFIXED.search(field):
            continue
        if field in FIELDS:
            meter, per, scale = FIELDS[field]
            found[meter] = (readable(value * scale), per)
            continue
        above = ABOVE.match(field)
        if above and above.group(1) in FIELDS:
            meter, per, scale = FIELDS[above.group(1)]
            found[f"{meter} above {int(above.group(2)) * 1000}"] = (
                readable(value * scale),
                per,
            )
    # A voice billed by the character in LiteLLM's output field alone is our characters rate;
    # with both fields, input is ours and output is shown beside it.
    if "characters" not in found and "characters (output)" in found:
        found["characters"] = found.pop("characters (output)")
    return found


def readable(usd: float) -> float:
    """Returns a figure to twelve significant digits, so a converted 0.042 reads 0.042."""
    return float(f"{usd:.12g}")


def ours(entry: dict[str, Json]) -> dict[str, tuple[float, int]]:
    """Returns an entry's vendor rates, by meter (or tier), as (usd, per)."""
    rates: dict[str, tuple[float, int]] = {}
    for meter, rate in (entry.get("rates") or {}).items():
        rates[meter] = (float(rate["usd"]), int(rate["per"]))
    for tier in entry.get("tiers") or []:
        for meter, rate in tier["rates"].items():
            rates[f"{meter} above {tier['above_input_tokens']}"] = (
                float(rate["usd"]),
                int(rate["per"]),
            )
    return rates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--list",
        type=Path,
        default=VENDOR,
        help="a price file, or router prices show --json",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="write JSON rows instead of tab-separated ones",
    )
    args = parser.parse_args()

    listed = json.loads(args.list.read_text())
    tag = latest_tag()
    catalog = fetch(CATALOG.format(tag=tag))
    if not isinstance(catalog, dict):
        raise SystemExit("LiteLLM's catalog is not an object")

    rows: list[dict[str, Json]] = []
    for entry in listed["entries"]:
        item = f"{entry['modality']}/{entry['provider']}/{entry['model']}"
        when = ",".join(
            f"{k}={v}" for k, v in sorted((entry.get("when") or {}).items())
        )
        key = match(catalog, entry["provider"], entry["model"])
        if key is None:
            rows.append({"item": item, "when": when, "litellm": "no match"})
            continue
        tier = (entry.get("when") or {}).get("service_tier", "")
        if tier and tier not in TIER_SUFFIXES:
            rows.append(
                {"item": item, "when": when, "litellm": f"{key}: no {tier} tier"}
            )
            continue
        theirs = figures(catalog[key], tier)
        mine = ours(entry)
        for meter in sorted(set(theirs) | set(mine)):
            row: dict[str, Json] = {
                "item": item,
                "when": when,
                "litellm": key,
                "meter": meter,
            }
            if meter in mine:
                row["ours_usd"], row["per"] = mine[meter]
            if meter in theirs:
                row["litellm_usd"], row["per"] = theirs[meter]
            if meter in mine and meter in theirs and theirs[meter][0]:
                row["gap"] = (mine[meter][0] - theirs[meter][0]) / theirs[meter][0]
            rows.append(row)

    if args.json:
        json.dump({"litellm_tag": tag, "rows": rows}, sys.stdout, indent=2)
        print()
        return
    print(f"# LiteLLM {tag}: item, when, LiteLLM key, meter, ours, LiteLLM, per, gap")
    for row in rows:
        gap = f"{row['gap']:+.2%}" if "gap" in row else ""
        print(
            "\t".join(
                str(value)
                for value in (
                    row["item"],
                    row["when"],
                    row["litellm"],
                    row.get("meter", ""),
                    row.get("ours_usd", ""),
                    row.get("litellm_usd", ""),
                    row.get("per", ""),
                    gap,
                )
            )
        )


if __name__ == "__main__":
    main()
