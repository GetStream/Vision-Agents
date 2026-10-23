---
name: check_prices
description: Check every price the router bills with against what each vendor charges today, and prepare the price changes for each one that is wrong. Use when a vendor announced new prices, before a release, or when asked whether our prices are right.
---

# Check prices

Every unit of work the router records is priced by one price list (`acceleration/internal/pricing`). Its rates are the vendors' list prices; an operator's markup, if any, is kept beside them and is never this skill's business.

| Where | What it is | How it changes |
| --- | --- | --- |
| `acceleration/prices/vendor.json` | The vendors' published list prices, embedded in the router. A new database is seeded from it at the router's first start | A PR. Editing it never changes a database that already has the item |
| A deployment's database (`price_changes`, `price_entries`) | The list once seeded, from `vendor.json` and any price file of the deployment's own (`ROUTER_PRICES_FILE`) laid over it. A router that does not yet price from the database prices from those two files | A new price change through the router's management API. A release's start only adds items the database never priced |

So a wrong price is fixed twice: a price change for each environment, and a PR to `vendor.json` so new deployments start right. This skill prepares both and writes neither to a shared environment.

## 1. What we charge

The vendor rates in force, read-only. Pick one:

- **The public list itself:** read `acceleration/prices/vendor.json`.
- **An environment's list:** in one of its router pods, `router prices show --json` (`kubectl exec … -- router prices show --json`). It reads the database and writes nothing. When the router serves the management API's price-list read (`GET /v1/manage/price-lists/active`), that answers the same: call it through `kubectl port-forward` to the router's Service with that environment's manage key and `X-Stream-Actor-Id: skill:check_prices`. Read the key inside a subshell from the environment's secret; never print it, log it or write it to a file, and let the port-forward stop when the subshell exits.
- **No cluster access:** start a router on a scratch database (the local Postgres the integration tests use, never a shared environment's); it seeds itself from `vendor.json`. Then `router prices show --json`.

`router prices show --unpriced` lists the routed models no entry prices: each is `missing`.

Skip the entries that are not a vendor's price: a model the routing config marks `local: true` (its rate, if any, estimates what running it costs the deployment), and the phone vendors' free `call`, `transfer` and `number` entries.

## 2. What vendors charge

Group the entries by provider. Dispatch one subagent per provider, in parallel, as `integrate_new_models` does. Each fetches the vendor's public pricing page and returns, per entry: the figure and unit as the page writes them, a short quote of the text around it, the URL and the date read. A page that needs a login, or is ambiguous about which model or tier a figure is for, is `unclear`: never guess.

Pages to start from. A subagent that finds one moved, or a provider missing, fixes this table in the same PR.

| Provider | Pricing page |
| --- | --- |
| `openai` | https://platform.openai.com/docs/pricing |
| `anthropic` | https://docs.anthropic.com/en/docs/about-claude/pricing |
| `gemini`, `google` | https://ai.google.dev/gemini-api/docs/pricing |
| `xai`, `grok` | https://docs.x.ai/docs/models |
| `deepgram` | https://deepgram.com/pricing |
| `elevenlabs` | https://elevenlabs.io/pricing/api |
| `cartesia` | https://cartesia.ai/pricing |
| `assemblyai` | https://www.assemblyai.com/pricing |
| `inworld` | https://inworld.ai/pricing |
| `together`, `together-*` | https://www.together.ai/pricing |
| `openrouter` | https://openrouter.ai/models (each model's page). Decision models are often not listed: check them with step 4b |
| `perplexity` | https://docs.perplexity.ai/getting-started/pricing |
| `deepseek` | https://api-docs.deepseek.com/quick_start/pricing |
| `exa` | https://exa.ai/pricing |
| `tavily` | https://www.tavily.com/pricing |
| `fal` | https://fal.ai/pricing (each model's page) |
| anyone else | the vendor's own docs; add the row |

## 3. A second opinion

```bash
uv run --no-project .claude/skills/check_prices/litellm.py
```

It reads LiteLLM's price catalog at its latest release tag (it prints the tag; never `main`) and prints its figures for the same items in our meters and units: `input_cost_per_token` × 1,000,000 is `input_tokens` per 1,000,000, `cache_read_input_token_cost` is `cached_input_tokens`, `cache_creation_input_token_cost` is `cache_write_input_tokens`, `*_above_<n>k_tokens` is a context tier above `<n>`,000, `input_cost_per_audio_token` is `input_audio_tokens`, `input_cost_per_second` × 3,600 is `audio_ms` per 3,600,000, `*_cost_per_character` is `characters` per 1,000,000, and `output_cost_per_image` is `images`. Pass `--list` a file from step 1 to check an environment's list, and `--json` for rows to work with. An item it cannot find reads `no match`.

A service-tier entry (`when: {"service_tier": …}`) is compared with LiteLLM's figures for that tier (its `_flex`, `_batches`, `_priority` and `_ultrafast` fields; `fast` is billed as `priority`). For a speech-to-speech model compare the per-modality meters (`input_audio_tokens`, `output_text_tokens` and the rest): its generic `input_tokens` and `output_tokens` are the fallback for a vendor that reports no split, at the audio rates, while LiteLLM's generic fields are the text rates.

Where the vendor's page and LiteLLM disagree, the page wins and the report says so. LiteLLM is never the source of a price.

## 4. Compare

Vendor rates with vendor rates, never with what a customer is charged, after converting units: per minute is `audio_ms` per 3,600,000 at 60×, per 1K characters is per 1,000,000 at 1,000×. Each entry is one of:

| Status | Meaning |
| --- | --- |
| `ok` | within half a percent |
| `under` | our vendor rate is below the vendor's: money lost on every request. Fixed first |
| `over` | above it: the customer overpays |
| `missing` | routed and unpriced (`--unpriced`) |
| `gone` | the vendor no longer lists the model |
| `unclear` | the page did not say |

Compare every tier too: each service-tier entry (`when: {"service_tier": …}`), each context tier, a cached or cache-write rate the vendor lists and the entry lacks (the list then charges those tokens the input rate: `over`), and a dated change the vendor announced that no scheduled change in the list carries yet.

## 4b. Against what the vendor billed

Some vendors say in their answer what the work cost them; the router keeps that figure on the row as `attributes.vendor_cost_micros`. In a router pod of the environment:

```bash
router prices drift --since 168h --json
```

It reads every shard and gives, per model, the number of such rows, the median gap between the list's cost at the vendor's rates and the vendor's figure (negative where the list billed less), and the ten rows with the widest gap. A median gap past 1% either way is `drift`: a wrong rate or a meter the list misses, never something to correct row by row. Report it with the command's example rows. For a model whose vendor publishes no price list, this is the only check there is.

## 5. Prepare the changes

For each vendor announcement, one price change as JSON in `.claude/skills/check_prices/out/<date>_<provider>.json` (git-ignored): never one per model, and never one for the whole run.

```json
{
  "name": "openai_gpt_price_change",
  "reason": "gpt-x input $2.00 → $2.50 per 1M tokens, cached $0.20 → $0.25",
  "source": "https://platform.openai.com/docs/pricing, read 2026-11-02",
  "effective_at": "2026-12-01T00:00:00Z",
  "entries": [
    {"modality": "llm", "provider": "openai", "model": "gpt-x",
     "rates": {"input_tokens": {"usd": 2.5, "per": 1000000}, "cached_input_tokens": {"usd": 0.25, "per": 1000000}}}
  ]
}
```

- An entry replaces the item's entry whole, so it carries every rate and tier the item has, not only the one that moved.
- `reason` says old → new with units for each model; `source` the URL and the date read; `effective_at` the vendor's date when it is in the future, else leave it out.
- A `gone` model is withdrawn (`"withdrawn": true`, one entry for each `when` it has) only when it is also gone from `router.yaml`; otherwise report it.
- Never a markup.

## 6. Check them

Each file's entries must build a list: put them in a price file and run `router prices check --file <path>`. When the router serves the management API's preview (`POST /v1/manage/price-lists/changes/preview`), send each body there too, on a router started locally on the scratch database or through the port-forward of step 1 (it writes nothing): its old and new vendor rates must be exactly the report's.

## 7. Hand over

- A draft PR updating `vendor.json` with the same vendor rates, per the `pr` skill, so new deployments start from them. Rates only: the PR never adds a markup.
- The change files, for an operator to apply through each environment's management API. The skill never writes a price to a shared environment.

## 8. Report

One table of every entry checked, with its status, then what was prepared. Put the run's date, the LiteLLM tag and the count per status in the PR description.
