# Vapi to accelerate

Sources checked 2026-10-16: Vapi docs (`docs.vapi.ai`, assistants, tools, knowledge
base) and the assistant API reference (`POST /assistant`).

## Source artifacts to ask for

- `GET /assistant/{id}` in full, plus squad membership if the assistant is in a squad.
- Tool objects (`GET /tool/{id}`) for every id in `model.tools` / `toolIds`, with server URLs.
- Knowledge base files (`GET /file`) behind any query tool.
- Phone number / SIP setup, `server` webhook config, `credentialIds`, analysis plans.

## Core config mapping

| Vapi | Accelerate |
| --- | --- |
| `name` | `name` / `description` in `agent.yaml` |
| `model.messages[role=system].content` | `instructions.md` |
| `firstMessage` + `firstMessageMode` | `greeting: {text: ..., mode: exact}`; `assistant-waits-for-user` means no greeting |
| `{{variable}}` prompt slots / `variableValues` | Fetched by a tool or stated once; slots do not exist. Report unsourceable variables |
| `model.temperature`, `maxTokens`, emotion/stability settings | `Accelerated(max_tokens=...)` covers only the token cap; the rest is reported |
| `backgroundSound`, silence/timeout plans, `startSpeakingPlan`/`stopSpeakingPlan` | No equivalent keys; report |

## Voice and model pipeline

| Vapi | Accelerate |
| --- | --- |
| `transcriber: {provider: deepgram, model: nova-3, keywords}` | `stt: deepgram/nova-3`; keywords become `keyterms: [...]` |
| `model: {provider: openai, model: gpt-4.1-mini}` | `llm: openai/gpt-4.1-mini` |
| `voice: {provider, voiceId}` | `tts: <provider>/<model>` when the router serves it, plus `voice: <voiceId>`; an unserved provider means leave out and report |
| A realtime model as `model` (OpenAI Realtime / Gemini Live) | `sts: <provider>/<model>`; drop transcriber/voice keys |
| Model presets (Balanced etc.) | Nothing to map; the preset's three choices were already expanded above |
| Custom transcriber/voice endpoints | No equivalent unless the router serves that provider; report |

## Tool calling

`model.tools` entries, by `type`:

| Vapi tool | Accelerate |
| --- | --- |
| `function` with `server.url` | Python function with the same name/parameters making that call, or a connector |
| `function` without a server (client-handled) | Python function if the work is server-side; a browser/device action has no equivalent and is reported |
| `apiRequest` | Python function performing the request; static body fields become function constants, model-filled fields become parameters |
| `query` (knowledge base) | Not a tool here: files go to `knowledge/` and the tool is deleted |
| `mcp` | `mcp_servers: [{name, url, tools}]` in `agent.yaml`, or a catalog `connectors:` entry |
| `transferCall` | No tool; transfer policy in `instructions.md`, number in router phone setup. Report |
| `endCall`, `dtmf`, `voicemail` | No build needed / no equivalent; report each |
| `handoff` (squad member) | The target assistant becomes `skills/<name>.md`; the handoff tool disappears |

Tool `messages` (request-start/complete/failed narration) become one instruction about
narrating lookups, or are dropped and reported. Squad assistants never become separate
agents: one folder, one skill per squad member.

## Knowledge base

Query-tool knowledge bases: download the files into `knowledge/`; a base described only
by a provider/name with files in Vapi storage still needs the file contents, so ask for
them. There is no query-tool tuning (`topK`, score thresholds) to carry.

## Integrations and telephony

- `server.url` webhooks (status-update, end-of-call-report, tool-calls handled elsewhere):
  tool-calls tools map as above; status/report webhooks have no destination. Report.
- Phone numbers and SIP (`phoneNumber`, SIP trunk credentials) move to router phone
  setup (`acceleration/cmd/phone`); credentials are never copied into the folder.
- `analysisPlan` (summary, structured data, success evaluation): no equivalent
  structured post-call pipeline. Transcripts exist; the extraction does not. Report.
- `artifactPlan`, recording/transcript destinations: no equivalent; report.

## Memory and state

Assistant-scoped state does not exist across calls. `variableValues` defaults become
tool lookups or opening instructions. Client-side state (Web SDK) is out of scope and
reported.

## No equivalent: do not invent

Squads as deployable units, assistant versioning via `toolIds` pinning, voicemail
detection plans, keypad/DTMF input plans, background-denoising config, per-call
`assistantOverrides`, and Vapi-hosted phone numbers. List each in the report.

## Mini example

Source:

```json
{"name": "Northwind support", "firstMessage": "Hello, this is Northwind support. How can I help?",
 "transcriber": {"provider": "deepgram", "model": "nova-3", "keywords": ["Northwind"]},
 "model": {"provider": "openai", "model": "gpt-4.1-mini",
   "messages": [{"role": "system", "content": "You are Northwind support..."}],
   "tools": [{"type": "function", "function": {"name": "lookup_order_status",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}}]},
 "voice": {"provider": "vapi", "voiceId": "Elliot"}}
```

Migrated:

```yaml
name: northwind_support
description: Northwind support.
stt: deepgram/nova-3
llm: openai/gpt-4.1-mini
keyterms: [Northwind]
greeting:
  text: Hello, this is Northwind support. How can I help?
  mode: exact
```

```python
@agent.llm.register_function(description="Look up the status of an order")
async def lookup_order_status(order_id: str) -> str:
    ...
```

Report line: `voice.provider vapi` (voice `Elliot`) is not served by the router; backend
default TTS used.
