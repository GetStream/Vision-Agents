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

## The top five, and what each calls the same thing

| Option | OpenAI | Anthropic | Gemini | xAI Grok | DeepSeek |
| --- | --- | --- | --- | --- | --- |
| `instructions` | `instructions` | `system` | `systemInstruction` | first `system` message | first `system` message |
| `input` | `input` | `messages` | `contents` | `messages` | `messages` |
| `max_output_tokens` | `max_output_tokens` | `max_tokens` (required) | `generationConfig.maxOutputTokens` | `max_tokens` | `max_tokens` |
| `temperature` | `temperature` | `temperature` (0–1) | `generationConfig.temperature` | `temperature` | `temperature` |
| `reasoning_effort` | `reasoning.effort` | `thinking.budget_tokens` | `thinkingConfig.thinkingBudget` | `reasoning_effort` (low, high) | reasoner model instead |
| `format` | `text.format` | tool-shaped output | `responseMimeType`, `responseSchema` | `response_format` | `response_format` |
| `verbosity` | `text.verbosity` | — | — | — | — |
| `tools`, `tool_choice` | `tools`, `tool_choice` | `tools`, `tool_choice` | `functionDeclarations` | `tools`, `tool_choice` | `tools`, `tool_choice` |
| `store`, `previous_response_id` | both | — | — | — | — |
| `prompt_cache_key` | `prompt_cache_key` | `cache_control` breakpoints | implicit and explicit caching | — | automatic, by prefix |
| `metadata` | `metadata` | `metadata.user_id` | — | — | — |

What the table is saying:

- **Thinking is not one scale.** OpenAI takes words, Anthropic and Gemini take token budgets,
  Grok takes two of the four words, and DeepSeek makes it a different model. There is no honest
  common scale, which is why each model declares the words it answers to.
- **Caching is three unrelated mechanisms.** A key you choose, breakpoints you place, and a
  prefix match you get for free. `prompt_cache_key` is the first; the rest is the provider's.
- **Only OpenAI holds the conversation for you.** Which is why the whole conversation is sent
  every time: consecutive turns may be answered by different providers, and a conversation that
  lives in the caller survives a failover. `previous_response_id` is an optimisation on top.

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

## Adding a parameter

1. A field on `ResponseParams`, and on `options.LLM` plus `Merge` if a config should be able to
   default it.
2. The same field on `LlmOptions` in
   [`openapi.yaml`](../../../acceleration/api/openapi.yaml), then regenerate all three clients.
3. Send it in each provider that takes it, and add a `Capabilities` field plus a `Validate` case
   if the answer changes when it is ignored.
4. A test in [`llmtest`](../../../acceleration/internal/llm/llmtest) so every provider is held
   to the same behaviour.
