# Sierra AI to accelerate

Sources checked 2026-10-16: Sierra product and blog pages (`sierra.ai/product/agent-sdk`,
`sierra.ai/blog/agent-development-life-cycle`). Sierra's own docs (`docs.sierra.ai`) sit
behind sign-in, so nothing here quotes a Sierra file format: none is public.

Sierra agents are built with a declarative Agent SDK: goals, composable skills,
guardrails, knowledge, and systems integrations, packaged as versioned releases. The
concepts map unusually well onto an accelerate folder, but the source must come from
the team (SDK source if they will share it, otherwise the journey/skill descriptions).
Never reconstruct a Sierra config from memory of the marketing pages.

## Source artifacts to ask for

- The Agent SDK source, or exported journey/skill definitions, in full.
- Goals and guardrails per journey, including determinism requirements.
- Knowledge sources and how releases snapshot them.
- Systems integrations: each API the skills call, with auth model (not secrets).
- The conversation-simulation test suite and its assertions.

## Core config mapping

| Sierra | Accelerate |
| --- | --- |
| Agent goal | `instructions.md`, opening with the goal in the agent's own words |
| Journey / workflow definition | The routing logic in `instructions.md`: when this kind of request arrives, hand it to that skill |
| Brand tone, policy guardrails | `instructions.md` for tone; hard rules to `guardrail.md` |
| Greeting per channel | `greeting: {text: ..., mode: exact}` (voice) |

## Voice and model pipeline

Sierra abstracts the model away (agents survive model upgrades without code changes),
so there is normally nothing to map: leave `stt`/`llm`/`tts` (or `sts`) out and run on
backend defaults. If the team pinned a voice or a speech-to-speech requirement for the
phone channel, map that one decision (`sts: <provider>/<model>` or a cascaded triple)
and report the rest as intentionally unmapped.

## Tool calling

| Sierra | Accelerate |
| --- | --- |
| Composable skill (procedural knowledge) | `skills/<name>.md` with the same steps; this is the cleanest mapping of any platform here |
| Deterministic API interaction inside a skill | A `@agent.llm.register_function(...)` the skill calls, or a connector when the system is catalogued |
| Skill that needs explicit confirmation/determinism | The confirmation stays a named step in the skill text; note in the report that enforcement is instructional, not programmatic |
| Supervisor / triage skill | Routing sentences in `instructions.md`, not a separate agent |

Sierra's declarative control flow is stricter than a skill file. Where a journey
*guaranteed* an order of operations, the skill must state the order just as explicitly,
and the report should flag that the guarantee is now instructional.

## Knowledge base

Release-snapshotted knowledge becomes `knowledge/` documents (and `knowledge/urls.yaml`
for living web sources, accepting that they refresh rather than snapshot). The snapshot
semantics differ: an accelerate agent reads what is synced now. Report that difference
if the team relied on release-pinned knowledge.

## Integrations and telephony

Systems integrations become connectors (app or per-user) or Python functions, one per
system of record. Channels (chat, phone, SMS, messaging) become Stream Chat/call plus
`channels:` entries for SMS/WhatsApp lines; phone numbers move to router phone setup
(`acceleration/cmd/phone`).

## Memory and state

Sierra's customer memory assembled from systems of record becomes tool lookups and
user-keyed backend memory. Journey state mid-conversation becomes skill state carried
in the conversation; there is no typed state object to port.

## No equivalent: do not invent

Agent OS releases/versioning as a deployment unit, the ADLC regression gating,
deterministic execution guarantees, Sierra's model abstraction layer itself, and
anything behind the docs sign-in that the team did not provide. Evals can be rebuilt
as `simulations/*.yaml` from their simulation suite; offer that, do not promise parity.

## Mini example

Source (as described by the team; not a Sierra file):

```text
Goal: resolve order and returns questions.
Skills: triage -> order_status (call the orders API, confirm before any change) -> confirm.
Guardrail: never change an order without an explicit customer confirmation.
```

Migrated: goal and triage become `instructions.md`, the guardrail lands in both
`instructions.md` and the skill, and `skills/order_status.md`:

```markdown
---
description: look up an order and, only after explicit confirmation, change it
---
Ask for the order id and call lookup_order_status. Read the result back.
Do not change anything until the customer confirms in so many words.
```

```python
@agent.llm.register_function(description="Look up the status of an order")
async def lookup_order_status(order_id: str) -> str:
    ...
```

Report line: Sierra's deterministic confirmation is instructional here, not enforced.
