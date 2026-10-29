# Vendors the product depends on

Every account the product pays for or holds a key to, grouped by what it is for. Read
2026-10-28 from `accelerate`.

Vendors are of two types:

- **Core**: the platform is built on them and needs them to work. Interchangeable core
  vendors (two hosts for one model) are one group: the active one and the alternative.
- **Connector**: used only by connectors and channels, which are bring-your-own: each
  customer connects their own account. Optional, not part of the platform.

How to read a row:

- **Key**: the env var and the line that reads it. The router reads each key as a pool
  (`internal/keypool/keypool.go:55`: `NAME=k1,k2`, then `NAME_1`, `NAME_2`).
- **Left**: what runs out. **Read**: the read-only API that says how much is left, with
  its doc, or the dashboard. `audit.py` calls every API marked *script*.

Amounts are not kept here: run `audit.py` for them.

## Core

### Routed models (speech, text, voices, search, images, detection)

| Vendor | Used for | Key (read at) | Left | Read |
| --- | --- | --- | --- | --- |
| OpenAI | LLM, realtime STS | `OPENAI_API_KEY` (`internal/llm/openai/openai.go:36`) | prepaid credit; spend | *script*, spend this month with `OPENAI_ADMIN_KEY` ([costs](https://developers.openai.com/api/reference/resources/admin/subresources/organization/subresources/usage/methods/costs)); credit balance: dashboard only ([billing](https://platform.openai.com/settings/organization/billing/overview)) |
| Anthropic | LLM | `ANTHROPIC_API_KEY` (`internal/llm/anthropic/anthropic.go:43`) | prepaid credit; spend | *script*, spend this month with `ANTHROPIC_ADMIN_KEY` ([cost report](https://platform.claude.com/docs/en/api/admin-api/usage-cost/get-cost-report), amounts in cents); credit balance: dashboard only |
| Google AI Studio | Gemini LLM, TTS, STT, Live STS, images | `GOOGLE_API_KEY` (`internal/llm/gemini/gemini.go:25`) | Prepay credit: at 0 every key of the billing account gets HTTP 402 ([billing](https://ai.google.dev/gemini-api/docs/billing)) | dashboard only: AI Studio Billing, "Available credits"; no API in the billing doc |
| xAI | Grok LLM, STT, STS | `XAI_API_KEY` (`internal/llm/xai/xai.go:18`) | prepaid credit | *script* with `XAI_MANAGEMENT_KEY`: `GET management-api.x.ai/v1/billing/teams/{team}/prepaid/balance` ([billing](https://docs.x.ai/developers/rest-api-reference/management/billing): `total` is USD cents and a purchase is negative, so the script prints `-total / 100` USD); the team id comes from `GET api.x.ai/v1/api-key` with `XAI_API_KEY` |
| Deepgram | STT | `DEEPGRAM_API_KEY` (`internal/stt/deepgram/deepgram.go:91`) | prepaid balance per project | *script*: `GET /v1/projects/{id}/balances` ([doc](https://developers.deepgram.com/reference/manage/billing/list)); reads the balance with `DEEPGRAM_ADMIN_KEY`: set it in `.env` to read this account |
| ElevenLabs | TTS, STT | `ELEVENLABS_API_KEY` (`internal/tts/elevenlabs/recorder.go:70`) | characters this cycle | *script*: `GET /v1/user/subscription`, limit − count ([doc](https://elevenlabs.io/docs/api-reference/user/subscription/get)) |
| Cartesia | TTS, STT | `CARTESIA_API_KEY` (`internal/tts/cartesia/recorder.go:90`) | plan credits | dashboard only. `GET /usage/credits` returns credits used per bucket, not left, and needs an admin key ([doc](https://docs.cartesia.ai/api-reference/usage/credits)) |
| Inworld | TTS, STT | `INWORLD_API_KEY` (`internal/tts/inworld/inworld.go:187`) | plan | dashboard only; none found (web search 2026-10-28) |
| Fish Audio | TTS, voice cloning | `FISH_API_KEY` (`internal/tts/voices/fish.go:35`) | API credit | *script*: `GET /wallet/self/api-credit` ([doc](https://docs.fish.audio/api-reference/endpoint/wallet/get-api-credit)) |
| Speechify | TTS | `SPEECHIFY_API_KEY` (`internal/tts/speechify/speechify.go:110`) | characters, USD balance | dashboard only: Console Usage page ([api limits](https://docs.speechify.ai/docs/api-limits)) |
| AssemblyAI | STT | `ASSEMBLYAI_API_KEY` (`internal/stt/assemblyai/assemblyai.go:55`) | prepaid balance | dashboard only ([billing doc](https://www.assemblyai.com/docs/billing-and-pricing)) |
| Together | `together` LLMs, Parakeet and Nemotron STT | `TOGETHER_API_KEY` (`internal/llm/together/together.go:19`) | prepaid credit; at 0 the API stops ([doc](https://docs.together.ai/docs/billing-credits)) | dashboard only; `GET /v1/billing/balance` answers 404 |
| Meta | Muse STT, Muse Spark LLM (`api.meta.ai`) | `META_API_KEY` (`internal/llm/meta/meta.go:19`) | unverified | dashboard only ([dev.meta.ai](https://dev.meta.ai/)); no billing API found |
| Microsoft (Azure MAI) | STT | `AZURE_MAI_API_KEY` (`internal/stt/microsoft/microsoft.go:49`) | Azure subscription, postpaid | postpaid — check budget/billing alerts at https://portal.azure.com/ |
| Alibaba DashScope | Qwen STS | `DASHSCOPE_API_KEY` (`internal/sts/openairealtime/openairealtime.go:107`) | unverified | dashboard only |
| MiniMax | LLM | `MINIMAX_API_KEY`, `MINIMAX` (`internal/llm/minimax/minimax.go:23`) | prepaid balance, with email alerts | dashboard only: Account > Billing > Balance ([faq](https://platform.minimax.io/docs/faq/about-account)) |
| OpenRouter | decision models | `OPENROUTER_API_KEY` (`internal/decisionmodel/openrouter/openrouter.go:20`) | prepaid credit | *script*: `GET /api/v1/credits`, total − usage ([doc](https://openrouter.ai/docs/api/api-reference/credits/get-remaining-credits) asks for a management key: set `OPENROUTER_MANAGEMENT_KEY` in `.env` to read this account) |
| Perplexity | search, decision models | `PERPLEXITY_API_KEY` (`internal/search/perplexity/perplexity.go:31`) | prepaid credit; keys blocked at 0 | dashboard only: API console billing ([tiers](https://docs.perplexity.ai/docs/admin/rate-limits-usage-tiers)) |
| Typesafe | decision model (Jev) | `TYPESAFE_API_KEY` (`internal/decisionmodel/typesafe/typesafe.go:21`) | unverified | none found (https://typesafe.ai/) |
| Exa | search | `EXA_API_KEY` (`internal/search/exa/exa.go:31`) | credits | dashboard only; none found (web search 2026-10-28) |
| Tavily | search | `TAVILY_API_KEY` (`internal/search/tavily/tavily.go:29`) | plan credits this cycle (+ pay-as-you-go limit) | *script*: `GET /usage` ([doc](https://docs.tavily.com/documentation/api-reference/endpoint/usage)) |
| fal | image generation | `FAL_KEY` (`internal/imagegen/fal/fal.go:37`) | prepaid credit; requests refused under the lock threshold | *script*: `GET /v1/account/billing?expand=credits`, needs a key with `billing:usage:read` (READONLY, BILLING or FULL preset) ([doc](https://fal.ai/docs/platform-apis/v1/account/billing)) |
| Ollama Cloud | `ollama` LLM | `OLLAMA_API_KEY` (`internal/llm/ollama/ollama.go:28`) | included + purchased credits | *script*: `GET ollama.com/api/balance` ([doc](https://docs.ollama.com/api/balance)) |
| Moondream | vision LLM, detection | `MOONDREAM_API_KEY` (`internal/llm/moondream/moondream.go:21`) | monthly free credits, then pay as you go ([cloud](https://moondream.ai/cloud)) | dashboard only |
| Roboflow | detection | `ROBOFLOW_API_KEY` (`internal/detection/roboflow/roboflow.go:35`) | plan credits | SDK `Workspace.get_usage()` ([doc](https://docs.roboflow.com/developer/python-sdk/workspace-plan-and-usage)); HTTP path not verified |
| Ultralytics | detection (YOLO) | `ULTRALYTICS_API_KEY` (`internal/detection/ultralytics/ultralytics.go:33`) | free shared API, 20 requests a minute (`router.yaml`) | nothing to deplete |
| Cerebras | LLM | `CEREBRAS_API_KEY` (`internal/llm/cerebras/cerebras.go:22`) | unverified | dashboard only (https://cloud.cerebras.ai/) |

### Inference hosts the code supports

Each is an OpenAI-compatible LLM host with a key the code reads. Not audited.

| Vendor | Key (read at) |
| --- | --- |
| Fireworks | `FIREWORKS_API_KEY` (`internal/llm/fireworks/fireworks.go:19`) |
| DeepInfra | `DEEPINFRA_API_KEY` (`internal/llm/deepinfra/deepinfra.go:19`) |
| Novita | `NOVITA_API_KEY` (`internal/llm/novita/novita.go:19`) |
| SiliconFlow | `SILICONFLOW_API_KEY` (`internal/llm/siliconflow/siliconflow.go:22`) |
| CoreWeave | `COREWEAVE_API_KEY` (`internal/llm/coreweave/coreweave.go:20`) |
| io.net | `IONET_API_KEY` (`internal/llm/ionet/ionet.go:18`) |
| Parasail | `PARASAIL_API_KEY` (`internal/llm/parasail/parasail.go:19`) |
| Phala | `PHALA_API_KEY` (`internal/llm/phala/phala.go:19`) |
| Venice | `VENICE_API_KEY` (`internal/llm/venice/venice.go:18`) |
| Wafer | `WAFER_API_KEY` (`internal/llm/wafer/wafer.go:20`) |
| GMI Cloud | `GMI_API_KEY` (`internal/llm/gmicloud/gmicloud.go:21`) |
| Nextbit | `NEXTBIT_API_KEY` (`internal/llm/nextbit/nextbit.go:19`) |
| Inceptron | `INCEPTRON_API_KEY` (`internal/llm/inceptron/inceptron.go:15`) |
| AtlasCloud | `ATLASCLOUD_API_KEY` (`internal/llm/atlascloud/atlascloud.go:15`) |
| Baidu Qianfan | `QIANFAN_API_KEY` (`internal/llm/baidu/baidu.go:20`) |
| Sail Research | `SAIL_API_KEY` (`internal/llm/sailresearch/sailresearch.go:19`) |
| DigitalOcean | `DIGITALOCEAN_INFERENCE_KEY` (`internal/llm/digitalocean/digitalocean.go:18`) |
| Cloudflare Workers AI | `CLOUDFLARE_API_KEY` (`internal/llm/cloudflare/cloudflare.go:29`) |

### Memory, knowledge, sandboxes

| Vendor | Used for | Key (read at) | Left | Read |
| --- | --- | --- | --- | --- |
| Mem0 | agent memory | `MEM0_API_KEY` (`internal/memory/mem0/mem0.go:33`) | plan (unverified) | dashboard only (https://app.mem0.ai/); none found (web search 2026-10-28) |
| Turbopuffer | knowledge search | `TURBOPUFFER_API_KEY` (`internal/knowledge/turbopuffer/turbopuffer.go:34`) | unverified | dashboard only (https://turbopuffer.com/dashboard) |
| Daytona | code sandboxes | `DAYTONA_API_KEY` (`internal/sandbox/daytona/daytona.go:32`) | unverified | dashboard only (https://app.daytona.io/dashboard/billing); no billing endpoint found |
| IPinfo | caller location | `IPINFO_TOKEN` (`internal/locate/locate.go:25`) | monthly requests (429 past the limit) | dashboard only ([developers](https://ipinfo.io/developers)) |

### GPU and hosting

| Vendor | Used for | Source | Pays | Left | Read |
| --- | --- | --- | --- | --- | --- |
| Gemma GPU host: Nebius (active), Baseten (alternative) | Gemma: moved from Baseten in AI-1664/AI-1665, commit 91b440863 ("Put Gemma first again: it now runs on Nebius"); Baseten stays possible | `GEMMA_API_KEY` (`internal/llm/gemma/gemma.go:27`), `GEMMA_BASE_URL` (`internal/llm/gemma/gemma.go:35`) | us | prepaid balance and grants, debited in real time ([billing](https://nebius.com/services/billing)) | dashboard only (https://console.nebius.com/); no billing API found |
| Baseten | DeepSeek and GLM Model APIs; self-hosted Parakeet, Breeze, S2 Pro (`acceleration/deploy/*/README.md:1`); the alternative Gemma GPU host | `BASETEN_API_KEY` (`internal/llm/gemma/gemma.go:31`, `internal/stt/parakeet/parakeet.go:116`) | us | postpaid usage | *script*: spend this month, `GET /v1/billing/usage_summary` ([doc](https://docs.baseten.co/reference/management-api/billing/gets-billing-usage-summary-for-a-date-range)) |
| Railway | docs search service for agents | `sdks/js/tests/live/hosted.test.ts:52` | us | plan credits and usage | dashboard only; GraphQL `customerForUser` may carry credits (community answer, [API](https://docs.railway.com/reference/public-api)) |
| Google Cloud | the router's clusters, Postgres, Valkey, GCS buckets, Secret Manager, KMS (private repo runbook `infra/docs/accelerate_launch.md`) | — | us | postpaid | postpaid — check budget/billing alerts at https://console.cloud.google.com/billing |
| AWS | S3 release bucket (`.claude/skills/deploy/SKILL.md:24`); `s3://` voices bucket option (`acceleration/README.md:107`); Bedrock in the `aws` plugin | — | us | postpaid | postpaid — check budget/billing alerts at https://console.aws.amazon.com/billing/home#/budgets |
| Vercel | Volt dashboard, getstream.io docs (`vercel.json` in both repos) | — | us | plan usage | postpaid — check usage at https://vercel.com/dashboard |
| Stripe | the gateway's payments (`GATEWAY_STRIPE_SECRET_KEY`) | — | unverified | nothing to deplete | — |
| Stream | the video and chat apps agents join; `STREAM_API_KEY` | — | Stream itself | internal, no external bill | — |

Not used: ClickHouse and Datadog (no reference in `acceleration/` or the router chart);
Postgres and Redis run in Google Cloud when deployed, and from `compose.yaml:5,19` images
locally.

### Observability and CI

| Vendor | Used for | Source | Left | Read |
| --- | --- | --- | --- | --- |
| Sentry | router errors | `SENTRY_DSN`, read by the SDK (`acceleration/cmd/router/main.go:168`) | event quota of the plan ([quotas](https://docs.sentry.io/pricing/quotas/)) | dashboard only: https://sentry.io/settings/billing/ |
| GitHub Actions | CI; `runs-on: xcode-27` (`.github/workflows/client-sdks.yml:144`) | `.github/workflows/` | included minutes, then billed ([doc](https://docs.github.com/en/billing/managing-billing-for-your-products/managing-billing-for-github-actions/about-billing-for-github-actions)) | dashboard only: https://github.com/organizations/GetStream/settings/billing |
| CodeRabbit | PR review | `.coderabbit.yaml` | seats | dashboard only: https://app.coderabbit.ai/settings/subscription |
| GitGuardian | secret scanning | a GitGuardian API key | API quota | `GET api.gitguardian.com/v1/quotas` ([doc](https://api.gitguardian.com/docs#tag/Quota/operation/quotas)); not in the script |

Cloudflare needs no account here: phonecheck's `cloudflared tunnel --url`
(`.claude/skills/phonecheck/SKILL.md:215`) is a Quick Tunnel: "You do not need a
Cloudflare account or domain." ([doc](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/)).
Workers AI is in the inference hosts above.

## Connector

Bring-your-own: `audit.py` prints these rows under their own heading, and a balance
under `--min` here is marked but never fails the run.

### Telephony and SMS

| Vendor | Used for | Key (read at) | Left | Read |
| --- | --- | --- | --- | --- |
| Telnyx | phone numbers, calls, SMS | `TELNYX_API_KEY` (`internal/config/config.go:552`); phonecheck's test person reads `TELNYX_PERSON_API_KEY` (`cmd/phonecheck/person/telnyx/api.go:27`), "the account the person number is in" (`cmd/phonecheck/README.md:85`), set by whoever runs phonecheck; not audited | balance + credit limit | *script*: `GET /v2/balance` ([doc](https://developers.telnyx.com/api-reference/billing/get-user-balance-details)) |
| Twilio | phone numbers, calls | `TWILIO_AUTH_TOKEN` (`internal/phone/twilio/twilio.go:32`); phonecheck's test person reads `TWILIO_PERSON_AUTH_TOKEN` (`cmd/phonecheck/person/twilio/api.go:25`), "the account the person number is in" (`cmd/phonecheck/README.md:84`), set by whoever runs phonecheck; not audited | prepaid balance | *script*: `GET /2010-04-01/Accounts/{sid}/Balance.json` ([doc](https://www.twilio.com/docs/usage/api/account)) |
| Plivo | calls | `PLIVO_AUTH_TOKEN` (`internal/phone/plivo/plivo.go:36`) | `cash_credits` USD | `GET /v1/Account/{auth_id}/` ([doc](https://www.plivo.com/docs/account/api/account)); not in the script |
| Vonage | calls | `VONAGE_API_KEY` (`internal/phone/vonage/vonage.go:43`) | balance in EUR | `GET rest.nexmo.com/account/get-balance` ([doc](https://developer.vonage.com/en/api/account)); not in the script |
| DIDWW | numbers | `DIDWW_API_KEY` (`internal/phone/didww/didww.go:36`) | balance + credit | `GET /v3/balance` ([doc](https://doc.didww.com/api3/2026-04-16/inventory-resources/balance/get-balance.html)); not in the script |
| Sinch | calls | `SINCH_KEY_SECRET` (`internal/phone/sinch/sinch.go:40`) | unverified | not searched |
| Bird | calls | `BIRD_ACCESS_KEY` (`internal/phone/bird/bird.go:38`) | unverified | not searched |
| Bandwidth | calls | `BANDWIDTH_ACCOUNT_ID`, `BANDWIDTH_USERNAME`, `BANDWIDTH_PASSWORD`, `BANDWIDTH_SITE_ID`, `BANDWIDTH_APPLICATION_ID` (`internal/phone/bandwidth/bandwidth.go:40-43,46`, read at `:99-111`) | unverified | not searched |
| Linq | iMessage lines | a Linq API key the app supplies with each line (`internal/api/channels.go:264`) | unverified | not searched |
| Meta (WhatsApp) | WhatsApp lines | a Meta access token the app supplies with each line (`internal/channels/channels.go:71`) | unverified | not searched |

## Python SDK plugins: the customer's own key

`plugins/*` call vendors with the key of whoever runs the agent, read from that person's
environment. Not our bill. Vendors: Anam, Anthropic, AssemblyAI, AWS, Cartesia, Decart,
Deepgram, ElevenLabs, Fish, Gemini, Hugging Face, Inworld, MiniMax, Mistral, Moondream,
NVIDIA, OpenAI (also Baseten, DashScope), OpenRouter, Qdrant, Qwen (DashScope), Roboflow,
Sarvam, Speechify, Telnyx, Tencent, Turbopuffer, Twelve Labs, Twilio, xAI, and fal
(`wizper`). `fast_whisper`, `kokoro`, `local`, `pocket`, `smart_turn`, `ultralytics` and
`vogent` read no key. `getstream` and `stream` use Stream's own keys.

Mock, local and `local: true` models in `router.yaml` (`mock`, `parakeet`, `breeze`,
`s2pro`, `gemma`) bill no vendor per token; their GPUs are the Baseten and Nebius rows.
