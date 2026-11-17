# Cost tracking

[Sprint 4](../sprint4.md), "Cost tracking".

## Asked for

Every request — speech-to-text, text-to-speech and LLM routing, the full agent workflow, and
phone numbers — carries correct cost tagging, so a deployment serving several customers can
see what drives its spend. Tags are untyped labels the customer chooses:

```
cost_tracking={customer_id: 123, project: moderation, environment: dev}
```

Any keys they like.

## What exists

`routing.Tags` is a plain `map[string]string`, capped at sixteen keys per request, carried on
every row a session produces. A session is opened with the tags once and every row it writes
inherits them, so tagging cannot be forgotten halfway through a conversation. `cmd/agent` and
`cmd/phone` take `-tag key=value`, repeatable, through
[tagsflag.go](../../acceleration/internal/routing/tagsflag.go).

Coverage is the four things the sprint listed, plus two it did not:

| What                    | Row                                              |
| ----------------------- | ------------------------------------------------ |
| A transcribed turn      | `requests`, modality `stt`                        |
| A synthesis             | `requests`, modality `tts`                        |
| A completion            | `requests`, modality `llm`                        |
| A memory call           | `requests`, modality `memory`                     |
| A number bought         | `requests`, modality `phone`, plus `phone_numbers` |
| A delegated task        | `requests`, modality `llm` — the subagent routes like anything else |

## Why tags get their own tables

`stats_tags_hourly` and `stats_tags_daily` hold one row per bucket, modality, customer and
*single* label, rather than tags becoming more columns on `stats_hourly`.

The reason is that a request carries a *set* of labels, not one more dimension. A request
tagged `{project: support, environment: dev}` belongs in both breakdowns, so it is unrolled
into a row for each. Asking "what does project=support cost me" is then one indexed read
rather than a scan over a JSON column.

- `GET /v1/{modality}/stats/tags?key=project` breaks spend down by a label.
- `GET /v1/{modality}/stats?tag=project:support`, repeatable, narrows the ordinary stats.

Filtering reads the raw request rows rather than a rollup, because a rollup bucket has already
forgotten which labels its requests carried.

## Where the numbers come from

Prices are config, not provider responses: each model in
[router.yaml](../../acceleration/internal/routing/router.yaml) declares what this deployment
is billed, and the recorder prices every row once, at write time. A model with no price reports
a cost of zero rather than a guess.

This is deliberate. A per-request price lookup would tie routing to a vendor's billing API,
and self-hosted models have no published price at all — for those the rate is the GPU hour
divided by measured throughput, which only the deployment knows.

## Not done

Nothing outstanding.
