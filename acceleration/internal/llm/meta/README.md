# Meta Muse Spark

`meta.New` serves `muse-spark-1.3` through `https://api.meta.ai/v1`, using the shared OpenAI-compatible streaming implementation. Set `META_API_KEY` on the server. The default registry exposes the provider as `meta`; add the model to the calling app's routing configuration to select `meta/muse-spark-1.3`.

This integration uses Meta's hosted API. It does not deploy weights to Baseten and is not an internal-only/incognito route. It accepts only the verified standard model, not the contributor variant.

## Contract

- Text streams and final usage counts use the shared LLM event/response contract.
- Reasoning effort defaults to `low`. Verified/documented options exposed here are `minimal`, `low`, `medium`, `high`, and `xhigh`; `none` is not advertised because Meta's cookbook says availability is unreliable.
- Chat Completions counts reasoning tokens but does not expose reasoning text; `StreamsReasoning` is false.
- Tool choice accepts `auto` or the empty default. Other choices fail locally rather than silently changing tool-execution semantics. Omit offered tools when only prose is wanted.
- Tool calls, IDs, arguments and results replay through the shared conversation contract.
- Provider cancellation uses the shared stream lifecycle.
- Image input uses the existing image-capable chat adapter. Video, audio, Responses-only built-in tools and image generation are not exposed by this implementation.
- Server-managed conversation/storage features remain disabled in this Chat Completions adapter. That does not assert a provider-wide retention guarantee.

The provider is registered without changing existing default model/alias selections or inventing model prices. The calling application's routing configuration must specify its enabled model and verified billing configuration.

## Verification

```sh
go test ./acceleration/internal/llm/meta ./acceleration/internal/llm/openaicompat ./acceleration/internal/llmrouter
go test -tags integration ./acceleration/internal/llmrouter -run '^TestMetaRouterIntegrationSuite$' -count=1 -v
```

Run from the repository root. The integration suite uses `META_API_KEY` from the local environment and performs bounded real requests through `llmrouter.DefaultRegistry`. It checks streaming/usage, a tool-call/result round trip, and cancellation. It creates no VM or GPU deployment.

Sources: [Meta API fundamentals](https://github.com/meta-models/meta-model-cookbook/tree/main/01_api_fundamentals), [tool contract](https://github.com/meta-models/meta-model-cookbook/blob/main/01_api_fundamentals/03_tool_calling.ipynb), [reasoning contract](https://github.com/meta-models/meta-model-cookbook/blob/main/01_api_fundamentals/06_reasoning_tokens.ipynb).
