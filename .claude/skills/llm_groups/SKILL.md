---
name: llm_groups
description: Rank the models in every LLM shortcut (llm-fast, llm-smart, llm-thinking, llm-flow and the rest) by a per-group formula over measured latency, tool and JSON reliability, Artificial Analysis intelligence and price, and write a web report and JSON saying whether each group's preferred model should change. Use when asked whether a group has the right preferred model, where a new LLM belongs, to re-order or audit the LLM router, or after adding or re-pricing a model.
---

# LLM groups report

A shortcut in [`router.yaml`](../../../acceleration/internal/routing/router.yaml) routes over
every model that meets its requirements. One member may be `prefer:`red, and gets every
request while it is healthy; the rest are failover ordered by live health alone. So
"re-ordering" a group means choosing its preferred model (and sometimes its `only:` list or
a model's effort). This skill says which member each group's own criteria put first.

| File | What it is | Who changes it |
| --- | --- | --- |
| `groups.yaml` | Per group: where it is used, a plain explainer, gates, weighted criteria | A person, when what a group is for changes |
| `inventory.json` | Committed measurements: suite results per model and effort, Artificial Analysis numbers per effort, notes | `llmgroups bench` and `fetch_aa.py` |
| `report.py` | Ranks and writes `out/report.html` and `out/report.json` (git-ignored) | |
| `fetch_aa.py` | Artificial Analysis Intelligence Index per effort into the inventory | |
| [`acceleration/cmd/llmgroups`](../../../acceleration/cmd/llmgroups) | `snapshot`: groups and members from router.yaml with the router's own matching. `bench`: the task suite, through the router's own registry | |

Prices are never copied: `report.py` reads `acceleration/prices/vendor.json` (the `check_prices`
skill keeps it right). Membership is never copied: it comes from `snapshot`.

## 1. Keys

The suite calls the vendors, so it needs the team's local keys. They live in Secret Manager;
write them to a private temporary file, never over the repo's `.env`, and delete it after:

```bash
source ~/.config/rocky/gcp.env
D=$(mktemp -d); chmod 700 "$D"
rocky agents local secrets create_env -f "$D/env"
# ... the bench commands below, then:
rm -rf "$D"
```

`rocky agents local secrets list -f <vendor>` shows what is stored, masked. A Gemini key
answering 402 is not ours: ours are postpaid on `stream-agents-gemini` (`provider_audit`).

## 2. Measure

From `acceleration/`, in a subshell so the keys go no further:

```bash
go build -o /tmp/llmgroups ./cmd/llmgroups
INV=../.claude/skills/llm_groups/inventory.json
(set -a; . "$D/env"; set +a; /tmp/llmgroups bench -models gemini/gemini-4-hydrogen -inventory $INV)
```

| What you want | Flags |
| --- | --- |
| A new model | `-models provider/model` |
| One group's members | `-group llm-fast` |
| Everything not yet measured on this suite | `-all -missing` |
| Everything again (a new suite version, or a month on) | `-all` |
| A model at several efforts, for its sweet spot | `-models … -efforts low,medium,high` |

Defaults: 3 runs of each task, 10 requests in flight across all models interleaved so no
vendor sees them all. The whole catalogue is about 800 calls, 12 minutes and a few dollars.
Run it as a background command: a full run outlives a foreground shell. Do not run
`fetch_aa.py` or edit the inventory while a bench runs: bench writes the file when it ends,
over anything written in between.

The suite (`cmd/llmgroups/suite.go`, version `v2`): a one-sentence spoken reply, a weather
tool call, a booking pulled out as JSON, three maths problems with known answers, and a
colour question about an image for models that read images. It records time to the first
token or tool-call byte (over every task, and as `ttft_live` over the reply, tool call and
JSON only, which is what the live groups rank on: a maths problem's long think is not what a
caller waits for), time to the whole answer, tokens, and a pass rate per kind. It runs
each model as router.yaml builds it; that effort is the model's `routed_effort`. A sweep's
other efforts are kept beside it for the report's sweep table and never ranked. Changing a
task, a check or the token budget is a new `suiteVersion`.

An all-errors run never replaces an earlier good one: an outage on the day says nothing
about the model. It is kept as a flag in the report instead, which is how an endpoint that
stopped existing surfaces.

## 3. Artificial Analysis

```bash
uv run --no-project .claude/skills/llm_groups/fetch_aa.py --unmapped
uv run --no-project .claude/skills/refresh_model_stats/aa_stats.py | grep -i "<model>"
uv run --no-project .claude/skills/llm_groups/fetch_aa.py --family "provider/model=Family Name"
uv run --no-project .claude/skills/llm_groups/fetch_aa.py
```

The family is the leaderboard name before the parenthesis. Map only what is the same
weights in the same mode: `Muse Spark Open` is not `muse-spark-1.3`, `DeepSeek V4.1 Flash
1017` is the Ollama model and not Baseten's earlier V4.1 Flash, and a `(Reasoning)` row is
not a model the router runs without thinking. Unmapped models fall back to router.yaml's
`benchmark.intelligence_index`. Intelligence is read at the effort the model is routed at; set
`aa.routed_as` when the leaderboard names that effort differently (Opus runs at the API's
default, which is AA's `medium`). The router.yaml index is meant to be at the effort the router sends. When the
two disagree by 3 or more the report flags it: fix router.yaml with `refresh_model_stats`.

## 4. Report

```bash
uv run .claude/skills/llm_groups/report.py
open .claude/skills/llm_groups/out/report.html
```

Per group: the verdict, the explainer and where it is used, the gates, the formula, a bar per
ranked member split into what each criterion contributed (★ is the preferred model), the
members not ranked and why, and the data table. Then the effort sweeps and every data
problem found (failing endpoints, missing prices, stale or disagreeing numbers).

The formula, from `groups.yaml`:

```
score = 100 × Σ(weight × normalised metric) / Σ(weight of the metrics the model has)
```

Each metric is normalised to 0–1 across the group's ranked members, best = 1; latency, price
and context on a log scale. A member needs `min_coverage` (80%) of the weight and every `required` metric to be ranked,
so a model with no intelligence score cannot win on speed and price alone.
Scores compare members of one group, never across groups.

## 5. Read it, as an agent

`out/report.json` has, per group, `verdict.action`:

| Action | Meaning | Do |
| --- | --- | --- |
| `keep` | Preferred is first, or within `keep_margin` (5) points | Nothing |
| `change_prefer` | `suggested_prefer` leads by more than the margin | Propose it, with the contributions that decide it |
| `investigate` | Preferred is not ranked: failing, past a gate, or missing data | Read its `excluded` reason first; it is usually a broken endpoint |
| `consider_prefer` | No preferred model; the formula's first is `suggested_prefer` | Propose pinning it only if the group serves real traffic |
| `none` | Informational, health-ordered group | Check no member is excluded for a reason that means it does not belong |
| `fix_config` / `measure` | Preferred is not a member / nothing is ranked | Fix router.yaml / run bench |

Never change `prefer:` on the report alone. A switch moves real traffic: say what moves,
check the suggested model's `flags`, its data policy and its price, and ask before editing
router.yaml. One run is 3 calls per task; a lead under about 10 points rests on little, so
re-run `-group <name>` before proposing it. A verdict on a group whose criteria you think are
wrong is a `groups.yaml` change, made separately and said so.

## 6. A new model

After the provider work (`integrate_new_models`, `router-llm`, `pricing`):

1. `bench -models provider/model`, and `-efforts` across its levels if it has several.
2. `fetch_aa.py --family …` then `fetch_aa.py`.
3. `report.py`, and report which groups it joins, where it ranks in each, and any verdict it
   changes.
4. Commit `inventory.json` with the change that routes the model.
