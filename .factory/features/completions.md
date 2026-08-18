# Completions

[Sprint 3](../sprint3.md), steps 1 to 4.

## Asked for

Work out how to use DeepSeek Flash on Baseten. Write a Go abstraction covering DeepSeek
Flash, Gemma 4 and OpenAI, with the usual router and stats on top, and a `llm-fast` shortcut.

## What exists

[internal/llm](../../acceleration/internal/llm) is the contract: `Start`, `Respond`,
`Interrupt`, `Events`, `Close`, plus `Provider`, `Model` and `Reasoning`.

| Provider                                                     | Endpoint                                        |
| ------------------------------------------------------------ | ----------------------------------------------- |
| [deepseek](../../acceleration/internal/llm/deepseek)         | Baseten's shared Model APIs, so no deployment    |
| [openai](../../acceleration/internal/llm/openai)             | OpenAI                                           |
| [gemma](../../acceleration/internal/llm/gemma)               | A dedicated Baseten vLLM deployment              |

All three speak OpenAI-compatible chat completions, so they share one implementation in
[openaicompat](../../acceleration/internal/llm/openaicompat) and differ only in base URL,
credentials and the extra fields they send.
[cmd/chat](../../acceleration/cmd/chat) asks one question or holds a conversation.

## Four decisions

- **The conversation lives in the caller.** Every request carries the whole history rather
  than a provider-side thread id, so consecutive turns can be served by different providers
  and a failover loses nothing. `Instructions` is separate from `Messages` so a retry cannot
  drop the system prompt.
- **Reasoning is off by default.** DeepSeek's models reason unless told not to, which spends
  the token budget and most of the latency before the first word of the answer — the wrong
  trade for a live conversation. The provider turns it off through the chat template;
  `Options.Thinking` turns it back on and the reasoning then arrives as `ReasoningDelta`,
  separate from the answer, with `Reasoning()` telling a caller to expect it.
- **Several completions run at once, and cancellation is per completion.**
  `Interrupt(completionIDs ...string)` abandons the named ones, or all of them when given
  none. Barge-in passes none; the [harness](harness.md) names one, which is what lets a
  delegated task be abandoned without stopping the sentence being spoken. A completion that
  already produced text still settles and still reports what it cost.
- **Tool calling is absent from the contract.** Standardising a tool schema across providers
  is a bigger question than routing needs answered, and `Client()` reaches the provider's own
  tool support. What the agent needed instead was [delegation](harness.md), which is a
  different shape: it does not block the reply on a result.

## Targets and prices

`llm-fast` means whichever model answers quickest, in whatever language: the shorthand for
when the only thing that matters is that the answer starts arriving. The four capability
shortcuts work here too, with `en-high-accuracy` reaching the quality tier.

| Model                             | Tier         | In     | Cached  | Out     |
| --------------------------------- | ------------ | ------ | ------- | ------- |
| `deepseek/DeepSeek-V4-Flash-0731` | low-latency  | $0.13  | $0.028  | $0.26   |
| `openai/gpt-5.6-luna`             | low-latency  | $0.20  | $0.02   | $1.20   |
| `gemma/gemma-4-E2B-it`            | low-latency  | $0.032 | -       | $0.16   |
| `deepseek/DeepSeek-V4-Pro-0813`   | high-quality | $1.32  | $0.132  | $3.96   |
| `openai/gpt-5.6-terra`            | high-quality | $2.00  | $0.20   | $12.00  |
| `openai/gpt-5.6-sol`              | high-quality | $5.00  | $0.50   | $30.00  |

Per million tokens. Cached prompt tokens are billed once, at the cached rate, not twice.
Gemma is self-hosted, so its rates are an estimate of what the deployment costs rather than a
published price. Sol uses OpenAI's canonical `gpt-5.6-sol` id with medium reasoning effort.

## Not done

**Gemma is not deployed.** [deploy/gemma-4](../../acceleration/deploy/gemma-4) is written and
validated but needs its own deployment, because Gemma is not on Baseten's shared Model APIs.
With `GEMMA_BASE_URL` unset the provider fails to build and routing moves on, so a shortcut
still resolves.
