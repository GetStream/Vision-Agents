# Gemma 4 E2B on Baseten

Google Gemma 4 E2B instruction-tuned, served by vLLM's OpenAI-compatible server so it can
back the self-hosted tier of the LLM router.

**Not yet pushed.** The recipe is written and reviewed but nobody has deployed it, so
`GEMMA_BASE_URL` is unset and the `gemma` provider fails to build. Routing moves to the next
candidate, so `llm-fast` and the capability shortcuts keep working. This is the same position
`deploy/s2-pro` is in.

## Why this one needs deploying at all

DeepSeek reaches us through Baseten's shared Model APIs, which host a fixed set of popular
models behind one endpoint and need no deployment. Gemma is not on that list, so serving it
means renting a GPU. That is the whole difference between the two providers.

## Licence

Gemma 4 is **Apache 2.0**, so there is no commercial-use question to settle the way there is
for S2 Pro next door. The Hugging Face repo is still gated behind Google's terms, though, so
fetching the weights needs an `hf_access_token` secret with the terms accepted on the account
that owns it.

## Deploy

```bash
truss push --promote
```

`truss push` alone creates a published deployment but leaves the `production` environment
pointing at the previous one, so `--promote` matters. To promote a deployment after the fact:

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

E2B is about 5 GB at bf16 including its per-layer embeddings, so it fits an L4 with most of
the 24 GB left for the KV cache. `--max-model-len 32768` caps context well below the model's
128K to keep that cache affordable; a conversation never needs 128K, and reserving it would
cost throughput. Raise it if a use case actually wants long context.

The larger variants want more card: E4B roughly doubles the weights, and 26B A4B and 31B want
an H100. E2B is the deliberate choice here — routing to a small self-hosted model only pays
off if the GPU is cheap.

## Prefix caching

`--enable-prefix-caching` is not an optimisation to review later. Every LLM request in this
service carries the whole conversation, because a conversation that lives in the caller
survives a failover. That means all but the newest message is identical to the previous turn's
prompt, which is exactly what prefix caching is for. It is also what the
`per_million_cached_input_tokens` rate in `internal/routing/router.yaml` is priced against.

## Tool calling

`--enable-auto-tool-choice --tool-call-parser pythonic` is what lets this deployment answer
with a tool call. vLLM otherwise accepts a `tools` array and ignores it, replying in prose,
which looks like a model that decided not to call anything rather than a server that was
never able to. The agent's `transfer` and `press` tools go through this path, so a call the
model wants to hand to a human depends on both flags being set.

`pythonic` is the parser Gemma wants: it writes calls as `[transfer(to="+15551234567")]`
rather than the JSON block the Hermes and Mistral parsers read.

## Test it

The Go provider's integration test covers this deployment and skips until it exists:

```bash
cd ../../ && go test -tags integration ./internal/llm/gemma/
```

Or by hand once `GEMMA_BASE_URL` is set:

```bash
cd ../../ && go run ./cmd/chat -target gemma/gemma-4-E2B-it -text "Say hello."
```
