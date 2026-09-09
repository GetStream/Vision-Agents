---
name: router-llm
description: What the LLM router can be asked for, what each provider calls it, and what it refuses to fake. Read before adding a response parameter or an LLM provider.
---

# Routing completions

The per-modality half of [router-interface](../router-interface/SKILL.md). The vocabulary is
[`ResponseParams`](../../../acceleration/internal/llm/params.go), and
[`options.LLM`](../../../acceleration/internal/options/options.go) is the subset a router config
holds as defaults. It is deliberately the same vocabulary rather than a second one for the same
things, so nothing is translated on the way through.

The router speaks OpenAI's Responses shape because that is what most of the field now speaks:
instructions separate from input, one stream per response, tool calls as events. Providers that
only offer chat completions are adapted in
[`openaicompat`](../../../acceleration/internal/llm/openaicompat).

There is only one path here. A completion is already whole by the time it is returned, so there
is no `recording()` — what the socket buys is the answer arriving as it is written.

## The top six, and what each calls the same thing

| Option | OpenAI | Anthropic | Gemini Flash | xAI Grok | Muse | DeepSeek |
| --- | --- | --- | --- | --- | --- | --- |
| `instructions` | `instructions` | `system` | `systemInstruction` | `instructions`, or a first `system` message | `instructions` | first `system` message |
| `input` | `input` | `messages` | `contents` | `input` | `input` | `messages` |
| `max_output_tokens` | `max_output_tokens` | `max_tokens` (required) | `generationConfig.maxOutputTokens` | `max_output_tokens` | `max_output_tokens` | `max_tokens` |
| `temperature` | dropped on reasoning models | 1.0, or 400 | 0–2, and don't | 0–2 | 0–2 | ignored while thinking |
| `reasoning_effort` | `reasoning.effort`, `none`–`max` | `output_config.effort`, `low`–`max` | `thinkingConfig.thinkingLevel`, `low`–`high` | `reasoning.effort`, `low`–`xhigh` | `reasoning.effort`, `minimal`–`max` | `thinking`, then `reasoning_effort` |
| `format` | `text.format` | `output_config.format` | `responseMimeType`, `responseSchema` | `text.format` | `text.format` | `response_format`, JSON but not a schema |
| `verbosity` | `text.verbosity` | — | — | — | 400 | accepted, then ignored |
| `tools`, `tool_choice` | auto, none, required | auto, any, tool, none | `functionDeclarations`, `functionCallingConfig.mode` | auto, none, required | `auto` and nothing else | auto, none, required, named |
| `store`, `previous_response_id` | both, kept 30 days | — | the Interactions API only | both, kept 30 days | both | — |
| `prompt_cache_key` | `prompt_cache_key` | `cache_control` breakpoints, four of them | implicit over 4k tokens, or a `cachedContents` resource | `prompt_cache_key`, which is `x-grok-conv-id` | `prompt_cache_key` | automatic, by prefix |
| cache TTL | `prompt_cache_options.ttl`, which is 30m | 5m or 1h | whatever the explicit cache was made with | — | `prompt_cache_retention`, as a hint | — |
| `metadata` | `metadata`, `safety_identifier` | `metadata.user_id` | `labels`, Interactions only | accepted, then ignored | `metadata` | `user_id` |

What the table is saying:

- **Thinking is not one scale, and it is mostly not optional any more.** The ladders run from
  three rungs to seven, and Gemini Flash, Grok and Muse cannot be told not to think at all:
  Muse takes `none` and answers 400, Gemini took `minimal` away in 3.8 Flash. OpenAI's 5.6
  family and DeepSeek still have an off switch, which is what a live tier is built on. There is
  no honest common scale, which is why each model declares the words it answers to.
- **Temperature is being withdrawn.** Anthropic rejects anything but 1.0 on models after Opus
  4.6, OpenAI's own migration guide says to remove it on reasoning models, Gemini takes it and
  warns that below 1.0 the model loops, and DeepSeek accepts it and ignores it while thinking is
  on. `Temperature` is a pointer for exactly this reason: saying nothing is the only portable
  thing left to say.
- **Caching is four unrelated mechanisms.** A key you choose, breakpoints you place, a prefix
  match you get for free, and a cache you create as a resource and point at. `prompt_cache_key`
  is the first; the rest is the provider's. TTL is not a duration anyone picks either — 30m or
  nothing, 5m or 1h, a hint, or no say at all — which is why `CacheTTLs` is a list of what a
  provider accepts rather than a number.
- **The conversation still lives in the caller, though it no longer has to.** OpenAI, Grok,
  Muse and Gemini's Interactions API will all hold it now; Anthropic and DeepSeek will not.
  Sending it whole every time is what makes a failover to one of those two work, and what lets
  consecutive turns be answered by different providers. `previous_response_id` is an
  optimisation on top.
- **Thinking does not survive the crossing.** Every provider that reasons wants its own
  reasoning handed back on the next turn: Anthropic's signed blocks, Gemini's thought signatures
  (a missing one is a 400 on a function call), Grok's encrypted content (omitting it is their
  own top cause of cache misses), Muse's encrypted replay, and DeepSeek's `reasoning_content`,
  which has to be replayed when tools are in play and left out when they are not. It is opaque,
  it is per provider, and it means nothing to the next one, so it belongs to the turn it came
  from and is dropped when the provider changes.

## What the router refuses to fake

Not by `supports:` — `options.LLM.Terms()` returns nothing, because every provider here speaks
the whole of the response parameters. It is
[`Capabilities`](../../../acceleration/internal/llm/capabilities.go) instead, checked before the
request goes out:

- A reasoning effort the model does not accept is an error. Not silently dropped, and not passed
  through either, because a provider's own answer to an unknown effort is a 400 halfway through
  a phone call.
- A verbosity the model does not accept is an error, where the model accepts any.
- `store` and `previous_response_id` mean nothing to a provider that does not report `Store`,
  and asking for both a conversation and a previous response is refused as a contradiction.
- A cache TTL a provider does not offer is dropped, since the request still works and only
  costs more.

The rule of thumb: refuse what changes the answer, drop what changes only the bill, and say
which in a comment.

The providers themselves are the argument for keeping this table. Grok 4.5 takes `xhigh` and
runs at `high` without saying so, and DeepSeek takes `text.verbosity` and does nothing with it.
Both are worse than a 400, because the request looks answered. Muse is the same problem one
level up: its `tool_choice` accepts only `auto`, so a request that says `required` is a
different set of candidates rather than a preference to pass along and hope for.

## Data policy and overwrites

Neither exists here yet, and both should. `options.STT` and `options.Search` carry a
`DataPolicy` and an `Overwrites` map, and every STT model in
[`router.yaml`](../../../acceleration/internal/routing/router.yaml) declares its
`data_handling`; `options.LLM` carries neither and the `llm:` section declares none. The shape
is the one the other two already use:

```yaml
llm:
  providers: [openai, gemini]
  data_policy:
    allow_training: false
    retention: none
  overwrites:
    gemini:
      thinking_budget: 2048
```

It matters more here than it does for audio. Muse encodes the training decision in the model
id, so `muse-spark-1.3-contributor` is a tenth of the price and a licence to train on both the
prompts and the answers, one suffix away from the model that is not. And a model name is not
a contract: the DeepSeek weights this deployment routes to are served by Baseten under
Baseten's terms, not the ones on DeepSeek's own platform, where the privacy policy claims
training rights and names no retention window. That is why `data_handling` is declared per model
in the router's config rather than per vendor in anybody's documentation.

Overwrites are where the leftovers go — Gemini's legacy `thinkingBudget`, Anthropic's
`thinking.type`, Grok's `search_parameters`, Muse's `search_context_size` — with each provider
parsing its own block into a typed struct so an unknown field is refused rather than dropped.

## Adding a parameter

1. A field on `ResponseParams`, and on `options.LLM` plus `Merge` if a config should be able to
   default it.
2. The same field on `LlmOptions` in
   [`openapi.yaml`](../../../acceleration/api/openapi.yaml), then regenerate all three clients.
3. Send it in each provider that takes it, and add a `Capabilities` field plus a `Validate` case
   if the answer changes when it is ignored.
4. A test in [`llmtest`](../../../acceleration/internal/llm/llmtest) so every provider is held
   to the same behaviour.
