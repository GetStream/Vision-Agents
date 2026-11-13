# /// script
# requires-python = ">=3.12"
# dependencies = ["pyyaml>=6"]
# ///
"""Rank every LLM group's members and write the report.

Run from the repo root:

    uv run .claude/skills/llm_groups/report.py

Reads router.yaml (through `go run ./cmd/llmgroups snapshot`), acceleration/prices/vendor.json,
this skill's inventory.json and groups.yaml. Writes out/report.html for people and
out/report.json for agents.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SKILL = Path(__file__).resolve().parent
REPO = SKILL.parents[2]
OUT = SKILL / "out"
STALE_DAYS = 30

# Metrics where less is better.
LOWER_BETTER = {
    "ttft_live_p50",
    "ttft_live_p95",
    "ttft_p50",
    "ttft_p95",
    "total_p50",
    "cost_per_call",
    "blended_price",
    "input_price",
}

LABELS = {
    "intelligence": "Intelligence",
    "ttft_live_p50": "First token p50",
    "ttft_live_p95": "First token p95",
    "ttft_p50": "First token p50 (all tasks)",
    "ttft_p95": "First token p95 (all tasks)",
    "total_p50": "Whole answer p50",
    "cost_per_call": "Cost per call",
    "blended_price": "Blended price",
    "input_price": "Input price",
    "pass_voice": "Voice reply",
    "pass_tool": "Tool call",
    "pass_json": "JSON",
    "pass_reasoning": "Reasoning",
    "pass_image": "Image",
    "context_window": "Context window",
    "languages": "Languages",
    "trains_on_data": "Trains on data",
}

COLOURS = {
    "intelligence": "#6f5bd3",
    "ttft_live_p50": "#1f9d8b",
    "ttft_live_p95": "#167a6c",
    "ttft_p50": "#1f9d8b",
    "ttft_p95": "#167a6c",
    "total_p50": "#4cb8a8",
    "cost_per_call": "#d98a1c",
    "blended_price": "#e0a64a",
    "input_price": "#b8711a",
    "pass_voice": "#3b82c4",
    "pass_tool": "#2563a8",
    "pass_json": "#5b9bd5",
    "pass_reasoning": "#8fb8e0",
    "pass_image": "#c2547a",
    "context_window": "#7a8b99",
    "languages": "#9aa8b4",
}


def load_snapshot() -> dict[str, Any]:
    result = subprocess.run(
        ["go", "run", "./cmd/llmgroups", "snapshot"],
        cwd=REPO / "acceleration",
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def load_prices() -> dict[str, dict[str, float]]:
    """US list rates per 1M tokens for each provider/model, the default service tier."""
    raw = json.loads((REPO / "acceleration/prices/vendor.json").read_text())
    prices: dict[str, dict[str, float]] = {}
    for entry in raw["entries"]:
        if (
            entry.get("modality") != "llm"
            or entry.get("region") != "us"
            or entry.get("byok")
            or entry.get("when")
        ):
            continue
        rates: dict[str, float] = {}
        for meter, rate in entry.get("rates", {}).items():
            rates[meter] = rate["usd"] * 1_000_000 / rate["per"]
        if entry.get("free"):
            rates = {"input_tokens": 0.0, "output_tokens": 0.0}
        prices[f"{entry['provider']}/{entry['model']}"] = rates
    return prices


@dataclass
class Model:
    name: str
    yaml: dict[str, Any]
    inventory: dict[str, Any]
    price: dict[str, float] | None
    routed_effort: str | None = None
    metrics: dict[str, float | str | None] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


def routed_measurement(inv: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    measured = inv.get("measured", {})
    effort = inv.get("routed_effort")
    if effort is None and len(measured) == 1:
        effort = next(iter(measured))
    if effort is None:
        return None, None
    m = measured.get(effort)
    if not m or m.get("errors") == m.get("calls"):
        return effort, None
    return effort, m


def intelligence(model: Model) -> tuple[float | None, str]:
    aa = model.inventory.get("aa", {})
    by_effort = aa.get("by_effort", {})
    # routed_as names the leaderboard row for a model the router sends no effort to, which
    # then runs at its vendor's default: Opus at medium.
    effort = aa.get("routed_as") or model.routed_effort
    if (
        effort
        and effort in by_effort
        and by_effort[effort].get("intelligence") is not None
    ):
        return float(
            by_effort[effort]["intelligence"]
        ), f"Artificial Analysis at {effort}"
    if len(by_effort) == 1:
        only, row = next(iter(by_effort.items()))
        if row.get("intelligence") is not None:
            return float(
                row["intelligence"]
            ), f"Artificial Analysis ({only}; the only row)"
    yaml_value = model.yaml.get("benchmark", {}).get("intelligence_index")
    if yaml_value:
        return float(yaml_value), "router.yaml"
    return None, ""


def build_models(
    snapshot: dict[str, Any],
    inventory: dict[str, Any],
    prices: dict[str, dict[str, float]],
) -> dict[str, Model]:
    today = dt.date.today()
    models: dict[str, Model] = {}
    for y in snapshot["models"]:
        name = y["name"]
        inv = inventory.get("models", {}).get(name, {})
        model = Model(name=name, yaml=y, inventory=inv, price=prices.get(name))
        effort, m = routed_measurement(inv)
        model.routed_effort = effort
        model.notes.extend(inv.get("notes", []))

        value, source = intelligence(model)
        model.metrics["intelligence"] = value
        if source:
            model.notes.append(f"Intelligence from {source}.")
        yaml_ii = y.get("benchmark", {}).get("intelligence_index")
        if (
            value is not None
            and yaml_ii
            and source.startswith("Artificial")
            and abs(yaml_ii - value) >= 3
        ):
            model.flags.append(
                f"router.yaml says intelligence {yaml_ii}, Artificial Analysis says {value:g} at the effort sent: run refresh_model_stats"
            )
        rows = inv.get("aa", {}).get("by_effort", {})
        if (
            source == "router.yaml"
            and rows
            and all(
                abs(r.get("intelligence", -99) - yaml_ii) > 1 for r in rows.values()
            )
        ):
            seen = ", ".join(f"{k} {r['intelligence']}" for k, r in rows.items())
            model.flags.append(
                f"router.yaml says intelligence {yaml_ii}, which matches no Artificial Analysis row ({seen}): find the effort it runs at and set aa.routed_as"
            )

        rates = model.price
        if rates is not None:
            model.metrics["input_price"] = rates.get("input_tokens")
            model.metrics["blended_price"] = (
                3 * rates.get("input_tokens", 0) + rates.get("output_tokens", 0)
            ) / 4
        elif not y.get("local"):
            model.flags.append("no vendor.json price")

        if m is not None:
            model.metrics["ttft_p50"] = m["ttft_p50_ms"]
            model.metrics["ttft_p95"] = m["ttft_p95_ms"]
            model.metrics["ttft_live_p50"] = m.get("ttft_live_p50_ms") or None
            model.metrics["ttft_live_p95"] = m.get("ttft_live_p95_ms") or None
            model.metrics["total_p50"] = m["total_p50_ms"]
            for kind, rate in m.get("pass_by_kind", {}).items():
                model.metrics[f"pass_{kind}"] = rate
            if rates is not None:
                fresh = max(m["input_tokens_mean"] - m["cached_tokens_mean"], 0)
                cached_rate = rates.get(
                    "cached_input_tokens", rates.get("input_tokens", 0)
                )
                model.metrics["cost_per_call"] = (
                    fresh * rates.get("input_tokens", 0)
                    + m["cached_tokens_mean"] * cached_rate
                    + m["output_tokens_mean"] * rates.get("output_tokens", 0)
                ) / 1_000_000
            if m.get("errors"):
                model.flags.append(
                    f"{m['errors']} of {m['calls']} suite calls failed: {m.get('error_sample', '')}"
                )
            age = (today - dt.date.fromisoformat(m["date"])).days
            if age > STALE_DAYS:
                model.flags.append(f"measured {age} days ago")
        elif effort is not None:
            sample = inv.get("measured", {}).get(effort, {}).get("error_sample", "")
            model.flags.append(f"every suite call failed at {effort}: {sample}")
        else:
            model.flags.append("never measured: run llmgroups bench")

        model.metrics["context_window"] = y.get("context_window") or None
        model.metrics["languages"] = y.get("languages") or None
        model.metrics["trains_on_data"] = (
            y.get("data_policy", {}).get("trains_on_data") or "unknown"
        )
        models[name] = model
    return models


@dataclass
class Ranked:
    model: str
    score: float
    coverage: float
    contributions: dict[str, float]


def normalise(values: dict[str, float], metric: str, scale: str) -> dict[str, float]:
    def tx(v: float) -> float:
        return math.log10(max(v, 1e-9)) if scale == "log" else v

    transformed = {k: tx(v) for k, v in values.items()}
    lo, hi = min(transformed.values()), max(transformed.values())
    if hi == lo:
        return dict.fromkeys(values, 1.0)
    out = {}
    for k, v in transformed.items():
        n = (v - lo) / (hi - lo)
        out[k] = 1 - n if metric in LOWER_BETTER else n
    return out


def gate_failure(model: Model, gate: dict[str, Any]) -> str | None:
    value = model.metrics.get(gate["metric"])
    label = LABELS.get(gate["metric"], gate["metric"])
    if value is None:
        return f"no {label} to check its gate against"
    if "max" in gate and isinstance(value, (int, float)) and value > gate["max"]:
        return f"{label} {fmt(gate['metric'], value)} is past the gate of {fmt(gate['metric'], gate['max'])}"
    if "min" in gate and isinstance(value, (int, float)) and value < gate["min"]:
        return f"{label} {fmt(gate['metric'], value)} is under the gate of {fmt(gate['metric'], gate['min'])}"
    if "equals" in gate and value != gate["equals"]:
        return f"{label} is {value!r}, not {gate['equals']!r}"
    return None


def rank_group(
    spec: dict[str, Any],
    members: list[str],
    models: dict[str, Model],
    defaults: dict[str, Any],
) -> tuple[list[Ranked], list[dict[str, str]]]:
    criteria = spec["criteria"]
    total_weight = sum(c["weight"] for c in criteria)
    min_coverage = spec.get("min_coverage", defaults["min_coverage"])

    excluded: list[dict[str, str]] = []
    eligible: list[str] = []
    for name in members:
        model = models[name]
        failure = next(
            (f for g in spec.get("gates", []) if (f := gate_failure(model, g))), None
        )
        if failure:
            excluded.append({"model": name, "reason": failure})
            continue
        missing = [
            c
            for c in criteria
            if not isinstance(model.metrics.get(c["metric"]), (int, float))
        ]
        have = total_weight - sum(c["weight"] for c in missing)
        if have / total_weight < min_coverage or any(
            c.get("required") for c in missing
        ):
            excluded.append(
                {
                    "model": name,
                    "reason": "needs data: "
                    + ", ".join(LABELS[c["metric"]] for c in missing),
                }
            )
            continue
        eligible.append(name)

    normalised: dict[str, dict[str, float]] = {}
    for c in criteria:
        values = {
            n: float(v)
            for n in eligible
            if isinstance(v := models[n].metrics.get(c["metric"]), (int, float))
        }
        normalised[c["metric"]] = (
            normalise(values, c["metric"], c.get("scale", "linear")) if values else {}
        )

    ranked = []
    for name in eligible:
        present = [c for c in criteria if name in normalised[c["metric"]]]
        weight = sum(c["weight"] for c in present)
        contributions = {
            c["metric"]: 100 * c["weight"] * normalised[c["metric"]][name] / weight
            for c in present
        }
        ranked.append(
            Ranked(
                name, sum(contributions.values()), weight / total_weight, contributions
            )
        )
    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked, excluded


def verdict(
    name: str,
    group: dict[str, Any],
    spec: dict[str, Any],
    ranked: list[Ranked],
    excluded: list[dict[str, str]],
    defaults: dict[str, Any],
) -> dict[str, Any]:
    prefer = group["prefer"]
    top = ranked[0] if ranked else None
    by_model = {r.model: r for r in ranked}
    margin = spec.get("keep_margin", defaults["keep_margin"])
    if not top:
        return {"action": "measure", "summary": "No member has enough data to rank."}
    if spec.get("informational") or not prefer:
        summary = f"Ordered by health. The formula would put {top.model} first ({top.score:.0f})."
        return {
            "action": "none" if spec.get("informational") else "consider_prefer",
            "suggested_prefer": top.model,
            "summary": summary,
        }
    if prefer not in group["members"]:
        return {
            "action": "fix_config",
            "summary": f"{prefer} is preferred but is not a member of {name}.",
        }
    if prefer not in by_model:
        reason = next((e["reason"] for e in excluded if e["model"] == prefer), "")
        return {
            "action": "investigate",
            "suggested_prefer": top.model,
            "summary": f"The preferred {prefer} is not ranked ({reason}). Top ranked: {top.model} ({top.score:.0f}).",
        }
    mine = by_model[prefer]
    position = ranked.index(mine) + 1
    if position == 1:
        return {
            "action": "keep",
            "summary": f"{prefer} ranks first ({mine.score:.0f}).",
        }
    gap = top.score - mine.score
    if gap <= margin:
        return {
            "action": "keep",
            "summary": f"{prefer} ranks {position} of {len(ranked)}, {gap:.0f} points behind {top.model}: within the {margin}-point margin.",
        }
    return {
        "action": "change_prefer",
        "suggested_prefer": top.model,
        "summary": f"{prefer} ranks {position} of {len(ranked)} ({mine.score:.0f}); {top.model} scores {top.score:.0f}, {gap:.0f} points ahead.",
    }


def fmt(metric: str, value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    if metric.startswith("ttft") or metric == "total_p50":
        return f"{value / 1000:.2f} s" if value >= 1000 else f"{value:.0f} ms"
    if metric == "cost_per_call":
        return f"${value * 1000:.2f}/1k"
    if metric in ("blended_price", "input_price"):
        return f"${value:.2f}"
    if metric.startswith("pass_"):
        return f"{value * 100:.0f}%"
    if metric == "context_window":
        return (
            f"{value / 1000:.0f}k" if value < 1_000_000 else f"{value / 1_000_000:.1f}M"
        )
    if metric == "intelligence":
        return f"{value:g}"
    return f"{value:g}"


def formula(spec: dict[str, Any]) -> str:
    terms = []
    for c in spec["criteria"]:
        arrow = "↓" if c["metric"] in LOWER_BETTER else "↑"
        scale = " (log)" if c.get("scale") == "log" else ""
        terms.append(f"{c['weight']:.2f}·{LABELS[c['metric']]}{arrow}{scale}")
    return "score = 100 × ( " + " + ".join(terms) + " ) ÷ weight present"


def e(text: Any) -> str:
    return html.escape(str(text))


def render(
    groups: list[dict[str, Any]], models: dict[str, Model], generated: str
) -> str:
    action_label = {
        "keep": "Keep",
        "change_prefer": "Change preferred",
        "investigate": "Investigate",
        "fix_config": "Fix config",
        "measure": "Measure",
        "consider_prefer": "Consider a preferred model",
        "none": "Health-ordered",
    }
    rows = []
    for g in groups:
        v = g["verdict"]
        top = g["ranking"][0]["model"] if g["ranking"] else "—"
        rows.append(
            f"<tr><td><a href='#{e(g['name'])}'>{e(g['name'])}</a></td><td>{e(g['prefer'] or '— health')}</td>"
            f"<td>{e(top)}</td><td><span class='pill {e(v['action'])}'>{e(action_label[v['action']])}</span></td></tr>"
        )

    sections = []
    for g in groups:
        spec = g["spec"]
        legend = "".join(
            f"<span class='key'><i style='background:{COLOURS[c['metric']]}'></i>{e(LABELS[c['metric']])} {c['weight']:.2f}</span>"
            for c in spec["criteria"]
        )
        bars = []
        for r in g["ranking"]:
            segments = "".join(
                f"<span class='seg' style='width:{points:.2f}%;background:{COLOURS[metric]}' title='{e(LABELS[metric])}: {points:.1f} points'></span>"
                for metric, points in r["contributions"].items()
            )
            star = " ★" if r["model"] == g["prefer"] else ""
            partial = (
                ""
                if r["coverage"] >= 0.999
                else f" <small>({r['coverage'] * 100:.0f}% of weight)</small>"
            )
            bars.append(
                f"<div class='bar{' preferred' if star else ''}'><div class='label'>{e(r['model'])}{star}</div>"
                f"<div class='track'>{segments}</div><div class='score'>{r['score']:.0f}{partial}</div></div>"
            )
        excluded = "".join(
            f"<li><b>{e(x['model'])}</b>: {e(x['reason'])}</li>" for x in g["excluded"]
        )
        metrics = [c["metric"] for c in spec["criteria"]]
        for extra in ("blended_price", "trains_on_data"):
            if extra not in metrics:
                metrics.append(extra)
        head = "".join(f"<th>{e(LABELS[m])}</th>" for m in metrics)
        body = []
        order = [r["model"] for r in g["ranking"]] + [x["model"] for x in g["excluded"]]
        for name in order:
            m = models[name]
            cells = "".join(f"<td>{e(fmt(k, m.metrics.get(k)))}</td>" for k in metrics)
            effort = e(m.routed_effort or "—")
            body.append(
                f"<tr><td>{e(name)}{' ★' if name == g['prefer'] else ''}</td><td>{effort}</td>{cells}</tr>"
            )
        gates = "".join(
            f"<li>{e(LABELS[x['metric']])} {e(gate_text(x))}: {e(x.get('why', ''))}</li>"
            for x in spec.get("gates", [])
        )
        sections.append(
            f"""
<section id="{e(g["name"])}">
  <h2>{e(g["name"])}{f" <small>{e(g['title'])}</small>" if g["title"] else ""}</h2>
  <p class="verdict {e(g["verdict"]["action"])}">{e(g["verdict"]["summary"])}</p>
  <p>{e(spec["explainer"])}</p>
  <p class="muted"><b>Used in:</b> {e(spec["used_in"])}</p>
  <p class="muted"><b>Preferred:</b> {e(g["prefer"] or "none, ordered by health")} · <b>Members:</b> {len(g["members"])}</p>
  {f"<ul class='gates'>{gates}</ul>" if gates else ""}
  <code class="formula">{e(formula(spec))}</code>
  <div class="legend">{legend}</div>
  <div class="chart" role="img" aria-label="Score of each ranked member of {e(g["name"])}">{"".join(bars) or "<p class='muted'>Nothing ranked.</p>"}</div>
  {f"<details><summary>Not ranked ({len(g['excluded'])})</summary><ul>{excluded}</ul></details>" if excluded else ""}
  <details><summary>Data</summary><div class="scroll"><table><thead><tr><th>Model</th><th>Effort</th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div></details>
</section>"""
        )

    flagged = [
        f"<li><b>{e(m.name)}</b>: {'; '.join(e(f) for f in m.flags)}</li>"
        for m in models.values()
        if m.flags
    ]
    sweeps = []
    for m in models.values():
        measured = m.inventory.get("measured", {})
        if len(measured) < 2:
            continue
        by_effort = m.inventory.get("aa", {}).get("by_effort", {})
        rows_ = "".join(
            f"<tr><td>{e(eff)}{' (routed)' if eff == m.routed_effort else ''}</td><td>{e(fmt('intelligence', by_effort.get(eff, {}).get('intelligence')))}</td>"
            f"<td>{e(fmt('ttft_p50', x.get('ttft_live_p50_ms')))}</td><td>{e(fmt('ttft_p50', x['ttft_p50_ms']))}</td><td>{e(fmt('ttft_p95', x['ttft_p95_ms']))}</td>"
            f"<td>{x['output_tokens_mean']:.0f}</td><td>{x['pass_rate'] * 100:.0f}%</td></tr>"
            for eff, x in measured.items()
        )
        sweeps.append(
            f"<h3>{e(m.name)}</h3><table><thead><tr><th>Effort</th><th>Intelligence</th><th>First token p50 (conversational)</th><th>p50 (all tasks)</th><th>p95 (all tasks)</th><th>Output tokens</th><th>Pass</th></tr></thead><tbody>{rows_}</tbody></table>"
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM groups report</title>
<style>
:root {{ --bg:#fff; --fg:#1d2330; --muted:#5d6676; --line:#e3e6eb; --track:#f1f3f6; --accent:#2563a8; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#14171c; --fg:#e6e9ee; --muted:#9aa3b2; --line:#2a2f37; --track:#1e232a; --accent:#7fb0e8; }} }}
body {{ font: 15px/1.5 -apple-system, system-ui, sans-serif; background:var(--bg); color:var(--fg); margin:0 auto; max-width:1100px; padding:24px; }}
h1 {{ margin:0 0 4px; font-weight:500; }} h2 {{ font-weight:500; margin:0 0 8px; }} h2 small {{ color:var(--muted); font-weight:400; }}
section {{ border-top:1px solid var(--line); padding:24px 0; }}
.muted, small {{ color:var(--muted); }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }} th, td {{ text-align:left; padding:4px 8px; border-bottom:1px solid var(--line); white-space:nowrap; }}
th {{ font-weight:500; color:var(--muted); }} .scroll {{ overflow-x:auto; }}
a {{ color:var(--accent); text-decoration:none; }}
.pill {{ padding:1px 8px; border-radius:10px; font-size:12px; background:var(--track); }}
.pill.change_prefer, .pill.investigate, .pill.fix_config {{ background:#f6d9a8; color:#5a3a00; }}
.pill.keep {{ background:#cfead9; color:#11502b; }}
.verdict {{ font-weight:500; }} .verdict.change_prefer, .verdict.investigate, .verdict.fix_config {{ color:#b26a00; }}
.formula {{ display:block; background:var(--track); padding:8px 10px; border-radius:6px; font-size:12.5px; margin:8px 0; white-space:pre-wrap; }}
.legend {{ display:flex; flex-wrap:wrap; gap:4px 14px; font-size:12px; color:var(--muted); margin:6px 0 10px; }}
.key i {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:-1px; }}
.bar {{ display:grid; grid-template-columns: minmax(140px, 300px) 1fr 70px; gap:10px; align-items:center; margin:3px 0; font-size:13px; }}
.bar .label {{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }} .bar.preferred .label {{ font-weight:600; }}
.track {{ display:flex; height:16px; background:var(--track); border-radius:3px; overflow:hidden; }} .seg {{ height:100%; }}
.score {{ font-variant-numeric:tabular-nums; }}
details {{ margin-top:10px; }} summary {{ cursor:pointer; color:var(--muted); }}
.gates {{ font-size:13px; color:var(--muted); margin:4px 0; }}
@media (max-width:640px) {{ .bar {{ grid-template-columns: 1fr 50px; }} .bar .track {{ grid-column: 1 / -1; grid-row: 2; }} }}
</style></head><body>
<h1>LLM groups</h1>
<p class="muted">Generated {e(generated)} from router.yaml, vendor.json and the llm_groups inventory. ★ marks each group's preferred model. Scores rank members within one group only.</p>
<table><thead><tr><th>Group</th><th>Preferred</th><th>Top by formula</th><th>Verdict</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
{"".join(sections)}
<section><h2>Effort sweeps</h2>{"".join(sweeps) or "<p class='muted'>None measured.</p>"}</section>
<section><h2>Data to fix</h2><ul>{"".join(flagged) or "<li>Nothing.</li>"}</ul></section>
</body></html>
"""


def gate_text(gate: dict[str, Any]) -> str:
    if "max" in gate:
        return f"at most {fmt(gate['metric'], gate['max'])}"
    if "min" in gate:
        return f"at least {fmt(gate['metric'], gate['min'])}"
    return f"must be {gate['equals']!r}"


def main() -> None:
    config = yaml.safe_load((SKILL / "groups.yaml").read_text())
    defaults = config["defaults"]
    inventory = json.loads((SKILL / "inventory.json").read_text())
    snapshot = load_snapshot()
    models = build_models(snapshot, inventory, load_prices())

    groups = []
    for group in snapshot["groups"]:
        spec = config["groups"].get(group["name"])
        if spec is None:
            print(
                f"warning: {group['name']} has no entry in groups.yaml; add one",
                file=sys.stderr,
            )
            continue
        ranked, excluded = rank_group(spec, group["members"], models, defaults)
        groups.append(
            {
                "name": group["name"],
                "title": group["title"],
                "prefer": group["prefer"],
                "members": group["members"],
                "spec": spec,
                "formula": formula(spec),
                "verdict": verdict(
                    group["name"], group, spec, ranked, excluded, defaults
                ),
                "ranking": [
                    {
                        "model": r.model,
                        "score": round(r.score, 1),
                        "coverage": round(r.coverage, 2),
                        "contributions": {
                            k: round(v, 1) for k, v in r.contributions.items()
                        },
                    }
                    for r in ranked
                ],
                "excluded": excluded,
            }
        )

    generated = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC")
    OUT.mkdir(exist_ok=True)
    (OUT / "report.html").write_text(render(groups, models, generated))
    machine = {
        "generated": generated,
        "groups": [
            {k: v for k, v in g.items() if k != "spec"}
            | {"criteria": g["spec"]["criteria"], "gates": g["spec"].get("gates", [])}
            for g in groups
        ],
        "models": {
            m.name: {
                "routed_effort": m.routed_effort,
                "metrics": m.metrics,
                "flags": m.flags,
                "notes": m.notes,
                "price_per_1m": m.price,
            }
            for m in models.values()
        },
    }
    (OUT / "report.json").write_text(json.dumps(machine, indent=2) + "\n")
    for g in groups:
        print(f"{g['name']:28} {g['verdict']['action']:16} {g['verdict']['summary']}")
    print(f"\nwrote {OUT / 'report.html'} and {OUT / 'report.json'}")


if __name__ == "__main__":
    main()
