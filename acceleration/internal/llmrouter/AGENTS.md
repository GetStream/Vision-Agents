# llmrouter

## OpenTelemetry GenAI names

The router follows the OpenTelemetry GenAI semantic conventions at
[`open-telemetry/semantic-conventions-genai@e6658f14c9d9`](https://github.com/open-telemetry/semantic-conventions-genai/blob/e6658f14c9d9ea0760825f029cb82e7673f48fab/docs/gen-ai/client-inference.md)
(2026-11-06). The conventions are in development and that repository has no release yet. To move to a newer
commit, change the names in `genai.go`, then this file and the test in `genai_test.go`.

### The span

Each model call gets one `chat <model>` span of kind `CLIENT` (`genai.go`). Each hedge attempt and each fallback is a
model call of its own. The span starts before the provider is asked. It ends when the response settles, at the same
point where the response's row is recorded. It sits beside `llm.provider.create`, under `llm.create`, so those spans
keep their parents. The span is exported only when tracing is on (`internal/tracing`).

| Attribute | Value |
| --- | --- |
| `gen_ai.operation.name` | `chat` |
| `gen_ai.provider.name` | The router's provider name, but `gcp.gemini` for `gemini` and `x_ai` for `xai`: the conventions' well-known values |
| `gen_ai.request.model` | The model of the provider config |
| `gen_ai.usage.input_tokens` | The prompt as the provider reported it, cached and cache-written tokens included |
| `gen_ai.usage.output_tokens` | The output as the provider reported it, reasoning included |
| `gen_ai.usage.cache_read.input_tokens`, `gen_ai.usage.cache_write.input_tokens`, `gen_ai.usage.reasoning.output_tokens` | Set only when non-zero |
| `error.type` | The row's error code (`create_failed`, `key_refused`, `provider_error`, `timeout`, `refused`) when the response did not run to its end. The span status is then `Error`. A response its caller closed (`cancelled`) is not an error |

The `gen_ai.usage.*` attributes carry only what the provider reported. When the provider reported nothing, the row is
billed on the router's own estimate and the span carries no token counts.

### Metrics

The router exports no OTel metrics. Its Prometheus metrics hold the same data under the names below. Their names and
labels stay as they are, because dashboards and monitors read them.

| GenAI metric | Router metric | What differs |
| --- | --- | --- |
| `gen_ai.client.inference.duration` | `router_request_seconds{modality="llm"}` | Labels `modality`, `provider`, no model. Buckets 25 ms to 60 s, not the convention's 10 ms to 81.92 s |
| `gen_ai.client.inference.time_to_first_chunk` | `router_first_output_seconds{modality="llm"}` | Labels `modality`, `provider`, `model`. Same buckets as above. `router_llm_stage_seconds` splits this time into stages and has no GenAI name |
| `gen_ai.client.inference.time_per_output_chunk` | none | No per-chunk timing is recorded |
| `gen_ai.client.inference.usage.input_tokens` | `router_tokens_total{modality="llm", kind="input"}` | The billed count: the router's estimate when the provider reported none |
| `gen_ai.client.inference.usage.cache_read.input_tokens` | `router_tokens_total{modality="llm", kind="cached_input"}` | As above |
| `gen_ai.client.inference.usage.cache_write.input_tokens` | none | Only on the span |
| `gen_ai.client.inference.usage.output_tokens` | `router_tokens_total{modality="llm", kind="output"}` | As above |
| `gen_ai.client.inference.usage.reasoning.output_tokens` | none | Only on the span |
| `gen_ai.client.inference.operation.input_tokens`, `.output_tokens` | none | No per-call token histogram |

The labels map like this. `provider` is `gen_ai.provider.name`, under the router's own name (`gemini`, `xai`).
`model` is `gen_ai.request.model`, but a `custom` provider's models are all counted as `custom`, so the label
stays bounded. `router_requests_total{outcome}` uses its own bounded set of outcomes, not `error.type`.
