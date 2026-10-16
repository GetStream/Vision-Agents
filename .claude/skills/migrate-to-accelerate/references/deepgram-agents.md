# Deepgram Voice Agent to accelerate

Sources checked 2026-10-16: Deepgram Voice Agent docs (`developers.deepgram.com/docs`,
configure-voice-agent, voice-agents-function-calling, voice-agent message flow).

A Deepgram agent is one websocket `Settings` message. There is no stored agent object:
the whole migration source is that JSON (plus whatever answers its function endpoints).

## Source artifacts to ask for

- The `Settings` message the client sends (`agent.listen/think/speak`, `audio`, `greeting`).
- For every `think.functions` entry with an `endpoint`: what answers it (code or URL).
- Client-side function handlers (functions without an endpoint).
- Conversation-history seeding and any `Update*` messages used mid-call.

## Core config mapping

| Deepgram | Accelerate |
| --- | --- |
| `agent.think.prompt` | `instructions.md` |
| `agent.greeting` | `greeting: {text: ..., mode: exact}` |
| Seeded history (`context`/history messages) | Durable facts into `instructions.md`; the rest dropped and reported |
| `audio.input/output` encodings and rates | Nothing; the backend owns media. Drop silently, it is transport, not behaviour |

## Voice and model pipeline

| Deepgram | Accelerate |
| --- | --- |
| `listen.provider {type: deepgram, model: nova-3, keyterms, language}` | `stt: deepgram/nova-3`; keyterms become `keyterms: [...]`; one language becomes `Accelerated(language=...)` |
| `listen` EOT thresholds/timeouts | No equivalent keys; the backend owns endpointing. Report |
| `think.provider {type: open_ai, model: gpt-4o-mini}` | `llm: openai/gpt-4o-mini` |
| `think` with a third-party `endpoint` (own LLM URL) | No equivalent custom-LLM endpoint unless the router serves that model; report |
| `speak.provider {type: deepgram, model: aura-2-...}` | `tts: deepgram/aura-2-...` when served, else leave out and report; a voice name is not a `voice:` id |
| `speak` with a non-Deepgram provider | Same rule: map when served, otherwise backend default and a report line |

The `think`/`speak`/`listen` split is exactly STT/LLM/TTS, so a Deepgram agent always
migrates as a cascaded agent, never `sts:`.

## Tool calling

`think.functions` entries:

| Shape | Accelerate |
| --- | --- |
| With `endpoint` (Deepgram calls your HTTPS URL) | Python function with the same name/parameters making that call, or a connector |
| Without `endpoint` (client-side; client answers `FunctionCallRequest`) | Python function if the work can run server-side; a device/browser action has no equivalent and is reported |
| `defer_until_eot: true` | An instruction to confirm before that action, noted in the report |

The function-call plumbing (`FunctionCallRequest`/`FunctionCallResponse`, ids) is
transport and disappears.

## Knowledge base

Deepgram Voice Agent has no knowledge base product; retrieval, if any, was one of the
functions. If a function only fetched documents, replace it with files in `knowledge/`
and delete the function. If it fetched live data, keep it as a function.

## Integrations and telephony

None in the Settings message: Deepgram agents are websocket audio in/out, so whatever
connected the call (telephony bridge, browser) is replaced by Stream call/chat. Any
bridging code is reported, not ported.

## Memory and state

Per-connection only. Nothing persists across Deepgram sessions either, so parity is
natural: user-keyed backend memory is a gain, not a mapping.

## No equivalent: do not invent

`UpdateThink`/`UpdateSpeak`/`UpdateListen` mid-call reconfiguration, injected agent/user
messages as an API, keep-alive and latency-report events, and custom think endpoints.
A prompt that changed mid-call becomes a skill boundary or a report line.

## Mini example

Source `Settings` (abridged):

```json
{"agent": {
  "listen": {"provider": {"type": "deepgram", "model": "nova-3", "keyterms": ["Northwind"]}},
  "think": {"provider": {"type": "open_ai", "model": "gpt-4o-mini"},
    "prompt": "You are Northwind support...",
    "functions": [{"name": "lookup_order_status", "description": "Look up the status of an order",
      "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}]},
  "speak": {"provider": {"type": "deepgram", "model": "aura-2-thalia-en"}},
  "greeting": "Hello, this is Northwind support. How can I help?"}}
```

Migrated:

```yaml
name: northwind_support
description: Northwind support.
stt: deepgram/nova-3
llm: openai/gpt-4o-mini
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

Report line: `speak.model aura-2-thalia-en` kept only if the router serves it; otherwise
the TTS field is left out rather than guessing a spelling.
