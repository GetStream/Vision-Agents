# Gemma 4 26B-A4B on Baseten

Google Gemma 4 26B-A4B instruction-tuned, thinking disabled, served by vLLM's
OpenAI-compatible server so it can back the self-hosted tier of the LLM router.

Live on Baseten as `gemma-4-26b-a4b-it` (`qvmmvrrq`), one H100, thinking off. Point
`GEMMA_BASE_URL` at the production OpenAI-compatible root:

```bash
GEMMA_BASE_URL=https://model-qvmmvrrq.api.baseten.co/environments/production/sync/v1
```

## Why this one needs deploying at all

DeepSeek reaches us through Baseten's shared Model APIs, which host a fixed set of popular
models behind one endpoint and need no deployment. Gemma is not on that list, so serving it
means renting a GPU. That is the whole difference between the two providers.

## Why 26B-A4B, not E2B or 12B

The live path wants the first word, not a think-then-speak turn. 26B-A4B is a Mixture-of-Experts
checkpoint: 26B total parameters, 4B active per token, so it decodes like a small dense model
while answering like a mid-size one. EAGLE3 speculative decoding on top of that is Baseten's
latency recipe. 12B dense would run every parameter on every token, which is slower to decode
than 4B-active MoE. E2B is cheaper to host and worse at the transfer/press tools the agent
actually needs.

Thinking stays off. For 26B that still emits an empty thought block, which the `gemma4`
reasoning parser strips so those tags never reach TTS.

## Licence

Gemma 4 is **Apache 2.0**, so there is no commercial-use question to settle the way there is
for S2 Pro next door. Google's Hugging Face repo is still gated behind their terms. This
recipe serves Red Hat's public FP8 checkpoint of the same instruct model, and fetching it
still needs an `hf_access_token` secret.

## Deploy

```bash
truss push --promote
```

`truss push` alone creates a published deployment but leaves the `production` environment
pointing at the previous one, so `--promote` matters. To promote a deployment after the
fact:

```bash
curl -X POST -H "Authorization: Api-Key $BASETEN_API_KEY" \
  https://api.baseten.co/v1/models/$MODEL_ID/deployments/$DEPLOYMENT_ID/promote
```

Then point `GEMMA_BASE_URL` at the OpenAI-compatible root, which is the environment's `sync`
path plus `/v1`:

```bash
GEMMA_BASE_URL=https://model-$MODEL_ID.api.baseten.co/environments/production/sync/v1
```

The Go provider appends `/chat/completions` to that, so the URL must end in `/v1` and nothing
more. `BASETEN_API_KEY` is the bearer token.

## No model.py

Unlike `deploy/parakeet` and `deploy/s2-pro`, this deployment has no Python. vLLM's image
already serves `/v1/chat/completions`, so `docker_server.start_command` runs `vllm serve`
directly and Baseten forwards to it. Nothing needs wrapping because the protocol we want is
the protocol the container speaks.

## GPU sizing

26B-A4B FP8 plus a 32K context cap fit a single H100, which is also the latency choice:
tensor parallel across two H100s is what Baseten's 256K preset uses, and a conversation never
needs that. `--max-model-len 32768` keeps the KV cache affordable. Raise it if a use case
actually wants long context.

## Prefix caching

`--enable-prefix-caching` is not an optimisation to review later. Every LLM request in this
service carries the whole conversation, because a conversation that lives in the caller
survives a failover. That means all but the newest message is identical to the previous turn's
prompt, which is exactly what prefix caching is for. It is also what the
`per_million_cached_input_tokens` rate in `internal/routing/router.yaml` is priced against.

## Tool calling

`--enable-auto-tool-choice --tool-call-parser gemma4` is what lets this deployment answer
with a tool call. vLLM otherwise accepts a `tools` array and ignores it, replying in prose,
which looks like a model that decided not to call anything rather than a server that was
never able to. The agent's `transfer` and `press` tools go through this path, so a call the
model wants to hand to a human depends on both flags being set.

## Test it

The Go provider's integration test covers this deployment and skips until it exists:

```bash
cd ../../ && go test -tags integration ./internal/llm/gemma/
```

Or by hand once `GEMMA_BASE_URL` is set:

```bash
cd ../../ && go run ./cmd/chat -target gemma/gemma-4-26B-A4B-it -text "Say hello."
```
