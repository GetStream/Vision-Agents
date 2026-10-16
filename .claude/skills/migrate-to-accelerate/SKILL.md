---
name: migrate-to-accelerate
description: Migrate a voice or text agent from LiveKit, Retell, Vapi, Pipecat, Decagon, Sierra AI, GPT Live, or Deepgram Agents to a Stream accelerate agent. Use when the user asks to migrate, port, convert, or rewrite an agent from one of those platforms, or hands over a competitor agent's code, dashboard export, or API object.
---

# Migrate to accelerate

An accelerate agent is a folder, not a vendor object:

```
<name>/
  agent.yaml        # name, description, mode, stt/tts/sts/llm/voice, greeting, channels
  instructions.md   # the system prompt
  guardrail.md      # optional: what the agent will and will not answer
  knowledge/        # documents and urls.yaml the agent looks things up in
  skills/<name>.md  # work handed to a slower, smarter model (frontmatter: description, deadline)
  simulations/      # optional: test conversations
  <name>.py         # Agent(config="<name>", llm=stream.Accelerated(config="<name>", ...))
```

`agent.yaml` refuses unknown keys. Only send keys from the mapping tables here and in the
reference for the source platform; never carry a competitor field name across hoping it works.

## First: read the right reference

Read exactly one of these before mapping anything, plus this file:

- LiveKit: [references/livekit.md](references/livekit.md)
- Retell: [references/retell.md](references/retell.md)
- Vapi: [references/vapi.md](references/vapi.md)
- Pipecat: [references/pipecat.md](references/pipecat.md)
- Decagon: [references/decagon.md](references/decagon.md)
- Sierra AI: [references/sierra.md](references/sierra.md)
- GPT Live: [references/gpt-live.md](references/gpt-live.md)
- Deepgram Agents: [references/deepgram-agents.md](references/deepgram-agents.md)

Then set up with the CLI skill at https://getstream.io/SKILL-agents.md (`getstream init`,
`getstream agents init --template voice|text`, `getstream docs` for anything not covered
here). For Stream Chat, Video, or Moderation work the migration touches, use the Stream
skills at https://www.skills.sh/getstream/agent-skills/stream.

## Workflow

1. **Inventory the source.** Get the whole agent, not the screenshot: code repo, dashboard
   export, or API object; every tool definition; knowledge sources; phone/SIP setup;
   webhooks; dynamic variables; guardrails and eval suites. Ask for what is missing.
   Vendor ids, secrets, and webhook signing keys are never copied into the output.
2. **Map config** with the shared table below and the platform reference. Write the folder:
   `getstream agents init <name> --template voice` (or `text`), then replace its files.
3. **Map tools** (rules below), **knowledge** (documents into `knowledge/`, URL sources
   into `knowledge/urls.yaml`), and **integrations** (catalog connectors and
   `mcp_servers` in `agent.yaml`; phone numbers and messaging lines under `channels:`).
4. **Sync and test.** `getstream agents sync <path>` then
   `echo "Where is order NW-1001?" | getstream agents test <name>` (stdin lines in a
   non-interactive terminal). Fix what the sync refuses; do not silence it by dropping
   fields silently.
5. **Report.** End with three lists: migrated, approximated (and how it differs), and
   no equivalent. A silent drop is a failed migration.

## Shared mapping

| Source concept | Accelerate target |
| --- | --- |
| Agent name, description | `agent.yaml` `name`, `description` |
| System prompt / instructions / general prompt | `instructions.md` |
| Who speaks first, first message | `greeting: {text: ..., mode: exact}` in `agent.yaml`; model-varied greeting is `mode: variation` |
| Cascaded pipeline: STT model | `stt: <provider>/<model>` in `agent.yaml`, or `Accelerated(stt=...)` |
| Cascaded pipeline: LLM | `llm: <provider>/<model>` or a capability shortcut such as `llm-fast`; or `Accelerated(model=...)` |
| Cascaded pipeline: TTS model + voice | `tts: <provider>/<model>`, `voice: <provider voice id>` |
| Speech-to-speech / realtime model | `sts: <provider>/<model>` (empty `sts: ""` takes the backend default); no separate STT/TTS |
| Function tool that runs your code | `@agent.llm.register_function(description=...)` in the Python entry point; the backend calls it back mid-conversation |
| Hosted/built-in tool (end call, transfer) | No tool needed for end/hangup; transfers and other platform actions go in `instructions.md` or a skill. See the platform reference |
| Procedure / workflow / AOP / state machine | A `skills/<name>.md` (description + instructions for the subagent), referenced from `instructions.md` |
| Tool that is an HTTPS call to a third-party system | A connector (`connectors:` in `agent.yaml`, app or per-user connection) or an `mcp_servers:` entry `{name, url, tools}`; a plain webhook-style tool becomes a Python function that makes the call |
| Knowledge base documents | Files in `knowledge/` |
| Knowledge base from URLs | `knowledge/urls.yaml` entries `{url, title?, description?, refresh_hours?}` |
| Retrieval tuning (top_k, chunk size) | No equivalent knob. Drop it and say so in the report |
| Knowledge in the prompt (inline docs) | Stays in `instructions.md` only if short; long text moves to `knowledge/` |
| Guardrails / content policy | `guardrail.md` |
| Phone number, SIP trunk | The router's phone setup (`acceleration/cmd/phone`; Twilio, Telnyx and other vendors in `acceleration/internal/phone`), not `agent.yaml` |
| WhatsApp / SMS / iMessage line | `channels: {whatsapp: {number: "+1..."}, identity: phone|link}` in `agent.yaml` |
| Memory / dynamic variables / user data | Session memory is keyed by user; per-user values arrive as session tags, not prompt variables. Map `{{var}}` prompt slots to instructions that ask or to tool results |
| Post-call analysis, summaries, webhooks on call end | Transcripts and episode cards on the router; a custom post-call webhook has no direct equivalent. Report it |
| Evals / simulations | `simulations/*.yaml`, run with `getstream agents simulate <name>` |
| Multilingual language setting | `Accelerated(language=...)`; keyterms lists map to `keyterms:` in `agent.yaml` |
| Sandbox / code execution | `sandbox: daytona` plus `sandbox_options:`; skills run code there |

Provider/model spellings: targets are `provider/model` names the router serves
(`deepgram/nova-3`, `openai/gpt-4.1-mini`, `cartesia/sonic-3`, `openai/gpt-live-1`).
When no model mapping is known, leave the field out (backend default) rather than
inventing a name, and note it in the report.

## Tool rules

- A Python function keeps its name, parameter names, and return text. JSON-schema tools
  (Vapi, Deepgram, Retell custom tools) become typed Python functions with the same
  schema in the type hints and docstring.
- Server-side tools with a URL: if the system has a connector, prefer it; otherwise the
  Python function performs the HTTPS call. The URL is configuration, not a secret; any
  credential it needed stays out of the folder and is named in the report as setup work.
- Client-side tools (browser/device actions) have no equivalent; the agent runs
  server-side. Report them.
- Handoffs between agents (LiveKit handoffs, Vapi squads, Retell transfer agents) become
  skills, not agents: one folder, one `skills/<name>.md` per specialist.
- Irreversible-action flags (`defer_until_eot`, strict/confirm modes) become an
  instruction to confirm before acting, plus a note in the report.

## Rules

- Never copy competitor code, keys, dashboard JSON verbatim, or signing secrets into the
  migrated folder. Re-express behaviour in the target shapes above.
- If a feature has no equivalent, say so in the report. Inventing an `agent.yaml` key,
  a model name, or a vendor API is a failed migration.
- Prefer the backend default over a guessed model. Prefer a skill over prompt stuffing
  for anything procedural. Prefer a connector over a hand-rolled HTTPS tool when the
  system is in the catalog.
- Keep the greeting, instructions tone, and tool behaviour identical unless the user
  asked to change them. Migration is not a rewrite.
