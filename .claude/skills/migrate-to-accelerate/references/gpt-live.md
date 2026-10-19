# GPT Live to accelerate

Sources checked 2026-10-16: OpenAI GPT-Live announcement and API docs
(`openai.com/index/introducing-gpt-live-1-in-the-api`,
`developers.openai.com/api/docs/models/gpt-live-1`,
`developers.openai.com/api/docs/guides/voice-websockets`).

GPT Live is OpenAI's full-duplex speech-to-speech model family (`gpt-live-1`,
`gpt-live-1-mini`), served over `v1/live/sessions` (WebRTC or WebSocket). It is not the
Realtime API: a session starts with `session.start`, and reasoning/tool work is
delegated to a backend Responses model.

The accelerate router serves GPT Live directly: `openai/gpt-live-1` is a speech-to-speech
target (`acceleration/internal/sts/openailive`), including delegation to a backend
Responses model. Most of a GPT Live agent therefore migrates as configuration, not redesign.

## Source artifacts to ask for

- The `session.start` payload: `session.model`, `instructions`, `audio`, `delegation`.
- The backend Responses config inside `delegation.responses`: model, instructions, `tools`.
- Any `session.update` usage, server-side controls, and WebRTC/WebSocket client code.
- What answers function tools, and any hosted tools (`web_search`, MCP) on the backend.

## Core config mapping

| GPT Live | Accelerate |
| --- | --- |
| `session.instructions` (short, conversational) | The opening of `instructions.md` |
| Backend instructions (Responses model) | The rest of `instructions.md`; depth belongs with the thinking model |
| `session.audio.output.voice` (e.g. `marin`) | `voice: <name>` only when the router accepts that voice for the target; otherwise leave out and report |
| `session.audio.format` | Nothing; transport. Drop |

GPT Live guidance keeps live instructions short because a backend does the thinking.
Keep that split: `instructions.md` holds both, and skills carry the heavy procedures.

## Voice and model pipeline

| GPT Live | Accelerate |
| --- | --- |
| `model: gpt-live-1` | `sts: openai/gpt-live-1` in `agent.yaml`; no `stt`/`tts`/`llm` keys for the voice path |
| `model: gpt-live-1-mini` | `sts: openai/gpt-live-1-mini` when served; confirm the exact spelling before writing it, else backend default plus a report line |
| `delegation.responses.model` | The subagent/skill model choice; name it in the relevant `skills/*.md` delegation, or report it as the backend-model decision for the config |
| Keyword biasing (`keyterms`-style hints) | `keyterms: [...]` in `agent.yaml` |

Do not split a GPT Live agent into STT+LLM+TTS to "preserve" it: full-duplex listening
while speaking is the point of the model, and `sts:` keeps it.

## Tool calling

Tools live on the delegated backend, not the voice model:

| GPT Live | Accelerate |
| --- | --- |
| Backend `tools: [{type: function, name, parameters}]` | `@agent.llm.register_function(description=...)` with the same name/parameters |
| Backend hosted `web_search` | Search is a router capability; keep the need in the skill instructions and report how it was wired |
| Backend MCP tools | `mcp_servers:` / `connectors:` in `agent.yaml` |
| Delegation mode settings, `tool_choice` | No equivalent keys; `tool_choice` is dropped and reported |

## Knowledge base

GPT Live has no KB product; retrieval was a backend tool or MCP server. Documents behind
a retrieval tool go to `knowledge/`; a live-data tool stays a function.

## Integrations and telephony

The Live session transport (WebRTC/WebSocket client, audio plumbing) is replaced by
Stream call/chat. OpenAI Presence deployments and account-level voice setup have no
equivalent and are reported. Phone reach, if any was bolted on, moves to router phone
setup (`acceleration/cmd/phone`).

## Memory and state

Session context the client managed (appended instructions, history replay) becomes
`instructions.md` plus user-keyed backend memory. `session.update` mid-call changes
become skill boundaries or report lines.

## No equivalent: do not invent

Full-duplex backchannel tuning, per-session delegation-mode switching, server-side
controls over someone else's live session, and ChatGPT Voice product behaviour. The
model capability itself (listen while speaking) carries across via `sts:`; its tuning
knobs do not.

## Mini example

Source `session.start` (abridged):

```json
{"type": "session.start", "session": {
  "model": "gpt-live-1",
  "instructions": "You are Northwind support. Keep replies short.",
  "audio": {"format": {"type": "audio/pcm", "rate": 24000}, "output": {"voice": "marin"}},
  "delegation": {"type": "responses", "responses": {"model": "gpt-5.6-luna", "tools": [
    {"type": "function", "name": "lookup_order_status", "description": "Look up the status of an order",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}]}}}}
```

Migrated:

```yaml
name: northwind_support
description: Northwind support.
sts: openai/gpt-live-1
```

```python
@agent.llm.register_function(description="Look up the status of an order")
async def lookup_order_status(order_id: str) -> str:
    ...
```

Report lines: backend model `gpt-5.6-luna` and voice `marin` are named decisions to
confirm against what the router serves; neither is silently substituted.
