# Retell AI to accelerate

Sources checked 2026-10-16: Retell API references (`docs.retellai.com/api-references`,
agent, Retell LLM, and knowledge base endpoints).

## Source artifacts to ask for

- `GET /get-agent/{id}` and its response engine: `GET /get-retell-llm/{id}`, or the
  conversation-flow / custom-LLM setup if it is not a Retell LLM.
- `general_tools` and state tools, with URLs, headers, and payload templates (not secrets).
- Knowledge bases: `GET /get-knowledge-base/{id}` sources (texts, URLs, files).
- Phone numbers bound to the agent, webhook URL, post-call analysis fields, dynamic variables.

## Core config mapping

| Retell | Accelerate |
| --- | --- |
| `agent_name` | `name` / `description` in `agent.yaml` |
| LLM `general_prompt` (+ `state_prompt` per state) | `instructions.md`; each state's prompt becomes a `skills/<state>.md` when states change behaviour, else fold into instructions |
| LLM `begin_message`, agent `begin_message` delay settings | `greeting: {text: ..., mode: exact}` |
| `start_speaker: user` | No greeting key; leave greeting out and say the agent waits |
| `default_dynamic_variables` (`{{var}}` in prompts) | Values fetched by a tool or stated once; `{{var}}` slots do not exist. Report any variable with no source |
| LLM `model`, `s2s_model` | Cascaded: `llm: <provider>/<model>`. Speech-to-speech (`s2s_model`): `sts: openai/<model>` and no STT/TTS |
| `model_temperature`, `tool_call_strict_mode`, `kb_config.top_k` | No equivalent keys; report |

## Voice and model pipeline

Retell agents do not name STT/TTS separately from the voice: `voice_id` (e.g.
`retell-Cimo`, an ElevenLabs/Cartesia voice) plus STT settings on the agent
(`stt_mode`, custom vocabulary) are the pipeline.

| Retell | Accelerate |
| --- | --- |
| `voice_id` | `voice: <id>` with a matching `tts:` provider/model when known; an unknown vendor voice means leave `tts`/`voice` out and report |
| Agent STT / `custom_stt_config` | `stt: <provider>/<model>` when the vendor is named; otherwise backend default |
| `language` / multi-language list | `Accelerated(language=...)` for one; report multi-language lists |
| `voice_speed`, `voice_temperature`, responsiveness/interruption settings | No equivalent keys; report |
| Ambient sound, backchannel, filler words | `Accelerated(backchannel=True)` covers backchannel only; the rest is reported |

## Tool calling

`general_tools` and per-state tools, by `type`:

| Retell tool | Accelerate |
| --- | --- |
| `custom` (URL, method, args schema, headers, query params) | Python function with the same name/args that makes the call, or a connector/`mcp_servers:` entry when the system is catalogued |
| `end_call` | Nothing to build; the backend ends calls. Drop and note it |
| `transfer_call` (predefined/cold/warm) | No tool equivalent. Put the transfer policy in `instructions.md`; the number itself is router phone setup. Report warm-transfer behaviour |
| `check_availability`, `book_appointment`, calendar tools | Connector for the calendar system when catalogued, else a Python function |
| `press_digit` / DTMF tools | No equivalent; report |
| State `edges` / state transitions | The state prompts become skills; edge conditions become skill instructions |

`speak_during_execution` / `speak_after_execution` filler text becomes an instruction to
narrate slow lookups, or is dropped and reported.

## Knowledge base

`knowledge_base_ids` on the LLM: export each base's sources. Texts and files become
documents in `knowledge/`; URLs become `knowledge/urls.yaml` entries (daily auto-refresh
maps to `refresh_hours: 24`). Chunk sizes and retrieval scores have no equivalent.

## Integrations and telephony

- Phone numbers (`/list-phone-numbers`) move to router phone setup
  (`acceleration/cmd/phone`); Retell-managed numbers do not port automatically.
- `webhook_url` (call started/ended/analyzed events) has no equivalent destination.
  Transcripts are stored by the router; a consumer of those webhooks needs new wiring.
  Report it, do not promise parity.
- `post_call_analysis_data` fields: no equivalent structured post-call extraction.
  Report; do not bury the fields in the prompt.

## Memory and state

Retell call-scoped dynamic variables and in-call state do not persist. Map variables to
tool outputs or session tags. Anything only a webhook used to supply must be fetched by
a tool instead.

## No equivalent: do not invent

Published versions and version pinning, agent environments/tags, call duration and
silence timers as config, denoising modes, guardrail/handbook config, post-call
analysis, and Retell's own SIP/telecom billing. List each in the report.

## Mini example

Source (Retell LLM + agent):

```json
{"general_prompt": "You are Northwind support...", "begin_message": "Hello, this is Northwind support.",
 "general_tools": [{"type": "custom", "name": "lookup_order_status", "url": "https://api.example.com/orders/status",
   "args": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}],
 "knowledge_base_ids": ["kb_northwind"], "model": "gpt-4.1-mini"}
```
```json
{"agent_name": "Northwind support", "response_engine": {"type": "retell-llm", "llm_id": "..."}, "voice_id": "retell-Cimo", "language": "en-US"}
```

Migrated:

```yaml
name: northwind_support
description: Northwind support.
llm: openai/gpt-4.1-mini
greeting:
  text: Hello, this is Northwind support.
  mode: exact
```

The prompt lands in `instructions.md`, the KB texts in `knowledge/`, and:

```python
@agent.llm.register_function(description="Look up the status of an order")
async def lookup_order_status(order_id: str) -> str:
    ...  # the same HTTPS call the Retell tool made
```

Report line: `voice_id retell-Cimo` has no known accelerate voice; backend default used.
