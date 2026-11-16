"""Pull Artificial Analysis Intelligence Index numbers into inventory.json, per effort.

Run from the repo root:

    uv run --no-project .claude/skills/llm_groups/fetch_aa.py
    uv run --no-project .claude/skills/llm_groups/fetch_aa.py --family "gemini/gemini-4-hydrogen=Gemini 4 Hydrogen"
    uv run --no-project .claude/skills/llm_groups/fetch_aa.py --unmapped

A model is mapped to an Artificial Analysis family name, the leaderboard name before the
parenthesis ("Gemini 4 Hydrogen" covers "Gemini 4 Hydrogen (Low)" ... "(Xhigh)"). Every row
of the family is stored under the effort in its parenthesis, lower-cased, with
"Non-reasoning" as "none" and a row with no parenthesis as "default".
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any

SKILL = Path(__file__).resolve().parent
sys.path.insert(0, str(SKILL.parent / "refresh_model_stats"))
from aa_stats import dicts, flight

INVENTORY = SKILL / "inventory.json"
VARIANT = re.compile(r"^(?P<family>.*?)\s*\((?P<variant>[^)]*)\)\s*$")


def effort_of(variant: str | None) -> str:
    if variant is None:
        return "default"
    first = variant.split(",")[0].strip().lower()
    return "none" if first == "non-reasoning" else first


def leaderboard() -> list[dict[str, Any]]:
    rows, seen = [], set()
    for value in flight("leaderboards/models"):
        for d in dicts(value):
            index = d.get("intelligenceIndex")
            if not isinstance(index, float) or d.get("slug") in seen:
                continue
            seen.add(d["slug"])
            speed = d.get("medianOutputTokensPerSecond")
            rows.append(
                {
                    "name": d["name"],
                    "slug": d["slug"],
                    "intelligence": round(index),
                    "output_tokens_per_second": round(speed)
                    if isinstance(speed, float)
                    else None,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family",
        action="append",
        default=[],
        help="provider/model=Family name, repeatable",
    )
    parser.add_argument(
        "--unmapped", action="store_true", help="list measured models with no family"
    )
    args = parser.parse_args()

    inventory = json.loads(INVENTORY.read_text())
    models = inventory.setdefault("models", {})

    for pair in args.family:
        name, _, family = pair.partition("=")
        models.setdefault(name, {}).setdefault("aa", {})["family"] = family

    if args.unmapped:
        for name, entry in sorted(models.items()):
            if not entry.get("aa", {}).get("family"):
                print(name)
        return

    rows = leaderboard()
    if not rows:
        sys.exit(
            "fetch_aa: the leaderboard returned no rows; the page shape changed (see refresh_model_stats)"
        )
    today = dt.date.today().isoformat()
    for name, entry in sorted(models.items()):
        aa = entry.get("aa", {})
        family = aa.get("family")
        if not family:
            continue
        by_effort = {}
        for row in rows:
            match = VARIANT.match(row["name"])
            row_family, variant = (
                (match["family"], match["variant"]) if match else (row["name"], None)
            )
            if row_family != family:
                continue
            by_effort[effort_of(variant)] = {
                k: row[k] for k in ("intelligence", "output_tokens_per_second", "slug")
            }
        if not by_effort:
            print(
                f"{name}: no leaderboard row named {family!r}; check the spelling with refresh_model_stats/aa_stats.py",
                file=sys.stderr,
            )
            continue
        aa["by_effort"] = dict(sorted(by_effort.items()))
        aa["read"] = today
        entry["aa"] = aa
        efforts = ", ".join(
            f"{k} {v['intelligence']}" for k, v in aa["by_effort"].items()
        )
        print(f"{name}: {efforts}")

    INVENTORY.write_text(json.dumps(inventory, indent=2) + "\n")


if __name__ == "__main__":
    main()
