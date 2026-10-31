# Gemma 4 26B-A4B on Baseten or Nebius

Google Gemma 4 26B-A4B instruction-tuned, thinking disabled, served by vLLM's
OpenAI-compatible server so it can back the self-hosted tier of the LLM router.

Any host that runs the vLLM recipe below works. The router needs two things from it: a root
URL that ends in `/v1`, and a bearer token. This directory covers two hosts:

| Host    | GPU                  | Status                                                          |
| ------- | -------------------- | --------------------------------------------------------------- |
| Nebius  | 1x H200, us-central1 | Serves staging since 2026-10-22 (AI-1665). |
| Baseten | 1x H100              | Model `gemma-4-26b-a4b-it` (`qvmmvrrq`). Alternative host. |

## Why this one needs deploying at all

DeepSeek reaches us through Baseten's shared Model APIs, which host a fixed set of popular
models behind one endpoint and need no deployment. Gemma is not on that list, so serving it
means renting a GPU on a host. That is the whole difference between the two providers.

## Why 26B-A4B, not E2B or 12B

The live path wants the first word, not a think-then-speak turn. 26B-A4B is a Mixture-of-Experts
checkpoint: 26B total parameters, 4B active per token, so it decodes like a small dense model
while answering like a mid-size one. EAGLE3 speculative decoding on top of that is the
latency recipe. 12B dense would run every parameter on every token. That is slower to decode
than 4B-active MoE. E2B is cheaper to host and worse at the transfer/press tools the agent
actually needs.

Thinking stays off. For 26B that still emits an empty thought block, which the `gemma4`
reasoning parser strips so those tags never reach TTS.

## Licence

Gemma 4 is **Apache 2.0**, so there is no commercial-use question to settle the way there is
for S2 Pro next door. Google's Hugging Face repo is still gated behind their terms. This
recipe serves Red Hat's public FP8 checkpoint of the same instruct model, and fetching it
still needs a Hugging Face token. On Baseten that is the secret `hf_access_token`.

## Deploy on Baseten

`config.yaml` is the Truss. Baseten mounts the weights from its Baseten Delivery Network
before the container starts, so a cold start does not wait on Hugging Face.

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

- GPU: `accelerator: H100` in `config.yaml`.
- `GEMMA_BASE_URL` is the environment's `sync` path plus `/v1`:

  ```bash
  GEMMA_BASE_URL=https://model-$MODEL_ID.api.baseten.co/environments/production/sync/v1
  ```

- Key: `BASETEN_API_KEY` is the bearer token. Leave `GEMMA_API_KEY` unset.

## Deploy on Nebius

Nebius runs the vLLM container as a Serverless AI endpoint. There is no Truss. We have not
replayed these steps. Each step below names its source. The live endpoint (AI-1664) exists,
but we have not read its arguments back.

Sources:
[Deploy a model](https://docs.nebius.com/serverless/tutorials/deploy-model),
[vLLM on Nebius](https://docs.vllm.ai/en/latest/deployment/frameworks/nebius/),
[Types of VMs and GPUs](https://docs.nebius.com/compute/virtual-machines/types),
[Manage endpoints](https://docs.nebius.com/serverless/endpoints/manage),
all opened 2026-10-28.

Set these first. Use a subnet in a project whose region offers H200: the live endpoint is in
`us-central1`, and the Nebius example uses `eu-north1`.

```bash
export SUBNET_ID=<subnet id>
export AUTH_TOKEN=$(openssl rand -hex 32)
```

The vLLM flags are the Baseten recipe in `config.yaml` under `docker_server.start_command`,
with the `/models/...` paths replaced by the Hugging Face repos from `weights`.

```bash
nebius ai endpoint create \
  --name gemma-4-26b-a4b-it \
  --image vllm/vllm-openai:v0.22.0-cu129 \
  --container-command vllm \
  --args "serve RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic \
    --served-model-name google/gemma-4-26B-A4B-it \
    --host 0.0.0.0 --port 8000 \
    --max-num-seqs 16 --max-model-len 32768 --gpu-memory-utilization 0.9 \
    --enable-prefix-caching --async-scheduling --trust-remote-code \
    --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
    --default-chat-template-kwargs '{\"enable_thinking\": false}' \
    --speculative-config.model RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3 \
    --speculative-config.num_speculative_tokens 3 \
    --speculative-config.method eagle3 \
    --limit-mm-per-prompt.image 0 \
    --load-format runai_streamer" \
  --platform gpu-h200-sxm \
  --preset 1gpu-16vcpu-200gb \
  --container-port 8000/http \
  --auth token --token "$AUTH_TOKEN" \
  --env-secret HF_TOKEN=<MysteryBox secret> \
  --shm-size 16Gi \
  --subnet-id "$SUBNET_ID"
```

Sources and status:

- Per Nebius docs (Deploy a model, vLLM on Nebius), not yet replayed by us: `--container-command vllm`,
  `--args` as one string that starts with `serve`, `--auth token --token`, `--container-port`.
- Per Nebius docs (Types of VMs and GPUs), not yet replayed by us: platform `gpu-h200-sxm`,
  preset `1gpu-16vcpu-200gb`, offered in `us-central1`.
- Per Nebius docs (Deploy a model), not yet replayed by us: `--env-secret KEY=SELECTOR` reads a MysteryBox
  secret. The docs do not name the key. `HF_TOKEN` is vLLM's usual name, and it is our guess.
- Not in the Nebius docs, not yet replayed by us: image `v0.22.0-cu129` (the docs pin `v0.19.1` by digest; we
  take `v0.22.0-cu129` from `config.yaml`), the swap of `/models/...` for Hugging Face repos
  with `--load-format runai_streamer`, and the flag set. Replay them on a test endpoint first.
- `--served-model-name google/gemma-4-26B-A4B-it` is required: the provider sends
  `google/gemma-4-26B-A4B-it` (`internal/llm/gemma/gemma.go`, `modelOwner`).
- Settings, per Nebius docs (Manage endpoints), not yet replayed by us: an endpoint cannot be changed after
  it is created. To change a flag, delete the endpoint and create a new one. The new one
  gets a new URL, so update `GEMMA_BASE_URL`.
- `GEMMA_BASE_URL` is the endpoint's `https://` public endpoint plus `/v1`
  (per Nebius docs, Deploy a model, not yet replayed by us; the `jq` filter is ours):

  ```bash
  ENDPOINT_ID=$(nebius ai endpoint get-by-name --name gemma-4-26b-a4b-it --format json \
    | jq -er '.metadata.id')
  ENDPOINT_URL=$(nebius ai endpoint get "$ENDPOINT_ID" --format json \
    | jq -er '[.status.public_endpoints[]? | select(startswith("https://"))] | .[0]')
  GEMMA_BASE_URL=${ENDPOINT_URL%/}/v1
  ```

- Key: `GEMMA_API_KEY` is the endpoint's token. `AUTH_TOKEN` above is that token. Treat it
  as a secret and never commit it. `BASETEN_API_KEY` is not the Nebius token, so set `GEMMA_API_KEY`.
- Check before you route to it: the URL appears while the endpoint is still `STARTING`, and
  `RUNNING` does not mean the model has loaded. Call `/health` and `/v1/models` first.

## Both hosts

The Go provider appends `/chat/completions` to `GEMMA_BASE_URL`, so the URL must end in
`/v1` and nothing more. Any other host that serves this vLLM recipe over OpenAI-compatible
chat completions works the same way. When `GEMMA_API_KEY` is unset, the provider sends
`BASETEN_API_KEY`.

## No model.py

Unlike `deploy/parakeet` and `deploy/s2-pro`, this deployment has no Python. vLLM's image
already serves `/v1/chat/completions`, so the host runs `vllm serve` directly: Baseten
through `docker_server.start_command`, Nebius through `--container-command` and `--args`.
Nothing needs wrapping because the protocol we want is the protocol the container speaks.

## GPU sizing

26B-A4B FP8 plus a 32K context cap fit a single GPU, which is also the latency choice:
tensor parallel across two GPUs is what Baseten's 256K preset uses, and a conversation never
needs that. `--max-model-len 32768` keeps the KV cache affordable. Raise it if a use case
actually wants long context. Baseten runs it on one H100 (`config.yaml`, where
`H100:2` is the step up for a longer context). Nebius runs it on one H200.

## Prefix caching

`--enable-prefix-caching` is not an optimisation to review later. Every LLM request in this
service carries the whole conversation, because a conversation that lives in the caller
survives a failover. That means all but the newest message is identical to the previous turn's
prompt, which is exactly what prefix caching is for. The Gemma entry in
`acceleration/prices/vendor.json` has no `cached_input_tokens` rate.

## Tool calling

`--enable-auto-tool-choice --tool-call-parser gemma4` is what lets this deployment answer
with a tool call. vLLM otherwise accepts a `tools` array and ignores it, replying in prose,
which looks like a model that decided not to call anything rather than a server that was
never able to. The agent's `transfer` and `press` tools go through this path, so a call the
model wants to hand to a human depends on both flags being set.

## Test it

The Go provider's integration test covers this deployment and skips unless `GEMMA_BASE_URL` and a key
(`GEMMA_API_KEY` or `BASETEN_API_KEY`) are set:

```bash
cd ../../ && go test -tags integration ./internal/llm/gemma/
```

Or by hand once `GEMMA_BASE_URL` is set:

```bash
cd ../../ && go run ./cmd/chat -target gemma/gemma-4-26B-A4B-it -text "Say hello."
```
