# Decagon to accelerate

Sources checked 2026-10-16: Decagon product pages (`decagon.ai/product`,
`decagon.ai/modules/knowledge-base`, `decagon.ai/product/integrations`) and
`docs.decagon.ai`.

Decagon is an enterprise deployment, not a self-serve API: there is no public agent
export to convert. A migration starts from what the team can describe (AOPs, knowledge
sources, integrations, escalation rules), usually gathered with the customer team.
Anything below marked as a Decagon concept is documented publicly; do not invent an
export format, field name, or API the team did not hand over.

## Source artifacts to ask for

- Agent Operating Procedures (AOPs): the natural-language procedures, in full text.
- Knowledge sources: help centre, Confluence/Contentful/Kustomer spaces, Snippets.
- Integrations list: CRM, helpdesk, CCaaS, commerce, and what actions each allows.
- Guardrails, tone/brand guidelines, QA criteria (Watchtower), escalation rules.
- Channels in use (chat, email, voice, SMS) and any per-channel behaviour differences.

If the team hands over screenshots instead of AOP text, that is a requirements
interview, not a migration: say so and ask for the text.

## Core config mapping

| Decagon | Accelerate |
| --- | --- |
| Agent persona, tone and brand guidelines | `instructions.md`; hard do/don't rules go to `guardrail.md` |
| Agent Operating Procedures | One `skills/<procedure>.md` per AOP (frontmatter `description` says when it applies); `instructions.md` routes to them |
| Greeting / opening message per channel | `greeting: {text: ..., mode: exact}` (voice); text channels open from instructions |
| Escalation procedure (when, to whom, with what summary) | A skill describing exactly that; the human destination is helpdesk wiring, reported as setup |

## Voice and model pipeline

Decagon does not expose STT/LLM/TTS choice as agent config the team can export. Voice
agents simply become a cascaded accelerate agent on backend defaults: leave `stt`,
`llm`, `tts` out unless the team names models they require. Model selection happening
inside Decagon has no equivalent and is not approximated.

## Tool calling

Decagon "actions" (look up an order, update an address, process a refund through an
integration) become:

| Decagon | Accelerate |
| --- | --- |
| Pre-built integration action (Zendesk, Salesforce, Shopify...) | `connectors:` entry for that system in `agent.yaml` (app connection), with the action kept as skill instructions |
| Custom endpoint action | Python function making the same call, or `mcp_servers:` if the system exposes MCP |
| An AOP step that only reads/decides | Stays inside the skill text; no tool |

Action inputs the AOP collects from the customer become the function's parameters or the
skill's questions. Approval/confirmation steps in an AOP stay as explicit skill steps.

## Knowledge base

This maps well, because the concepts match:

| Decagon | Accelerate |
| --- | --- |
| Synced help centre / docs sources | Export the articles into `knowledge/`; public URLs can go in `knowledge/urls.yaml` with `refresh_hours` |
| Snippets (internal short knowledge) | Documents in `knowledge/`, named for the topic |
| Suggestions / gap mining | No equivalent. Report; the migrated agent will not improve its own KB |
| Retrieval behaviour | Built in; no tuning to carry |

Get article exports, not portal access promises: a source nobody can export is a
report line, not a mapping.

## Integrations and telephony

- Helpdesk/CRM connections become connectors (app connections, connected once) plus the
  skill instructions that used them. Per-end-user lookups (a customer's own account)
  become session-binding connectors.
- Voice (CCaaS) and phone setup move to router phone setup (`acceleration/cmd/phone`).
- Chat/email widgets are replaced by Stream Chat channels; email as a channel has no
  equivalent in the folder and is reported.

## Memory and state

Customer context Decagon assembled from the CRM becomes tool lookups at conversation
start or per-user connectors. Conversation summaries Decagon wrote back to the helpdesk
have no automatic equivalent; report them.

## No equivalent: do not invent

Watchtower QA and monitoring, Duet's AOP generation/testing loop, Suggestions,
per-channel Decagon analytics, Decagon's model routing, and any Decagon API that was
not documented in what the team provided. The migration report should be explicit that
QA/evals need rebuilding as `simulations/` if wanted.

## Mini example

Source AOP (as a team would paste it):

```text
AOP: order_status. Ask for the order id, look it up in Shopify, read the status back,
offer tracking. If the order is older than 90 days, escalate to a human in Zendesk
with a summary.
```

Migrated `skills/order_status.md`:

```markdown
---
description: handle order-status questions by looking the order up in Shopify
---
Ask for the order id. Look it up ... (same steps, same 90-day rule, same escalation)
```

Plus a `connectors:` entry for Shopify in `agent.yaml`, the returns Snippet as
`knowledge/returns.md`, and report lines for Watchtower and Suggestions.
