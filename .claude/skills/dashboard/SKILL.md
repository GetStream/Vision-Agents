---
name: dashboard
description: How to work on the Volt dashboard (/Users/thierry/workspace/volt-dashboard), Stream's admin SPA, and its Agents section that talks to the acceleration router. Read before touching Volt or wiring a router change into it.
---

# Volt dashboard

Checkout: `/Users/thierry/workspace/volt-dashboard`. Not the same as this repo's `dashboard/`
(Next.js); Volt is Stream's production dashboard and the Agents section is being added to it.

Volt has its own rules. Read its `CLAUDE.md` and `.claude/rules/*.md` before editing, and
use its `.claude/skills/` (`new-route`, `new-api-hook`, `gen-api`, ...) for routine work.

## Stack

React 19, TypeScript 7, Vite, TanStack Router (file-based) + React Query, Tailwind 4,
Radix/shadcn. **Bun only**, never npm. Tests: Vitest + Testing Library (MSW for HTTP),
Playwright for e2e.

The shell defaults to Node 19, which breaks `bun lint`, `bun dev` and Vitest. Prefix every
Volt command with `nvm use 24 &&` (or `export PATH=~/.nvm/versions/node/v24.11.1/bin:$PATH`).

## Run it against a local router

Full guide: `docs/local-agents/README.md`. Short version:

```bash
# in Vision-Agents
export VOLT_CHECKOUT=/Users/thierry/workspace/volt-dashboard
docker compose -f compose.yaml -f "$VOLT_CHECKOUT/docs/local-agents/compose.volt.yaml" up -d --build router
# in volt-dashboard (.env.local: HOST=local.getstream.io PORT=3011 AGENTS_ROUTER_URL=http://localhost:8080)
bun dev
```

Open `https://local.getstream.io:3011/organization/<orgId>/<appId>/agents/` (not
`localhost`, or CORS/auth break). The override puts the router in `proxy` mode, so it reads the app and organization
the dev proxy sets, with Volt's
CORS origin. Rebuild the router after pulling router changes.

For connector OAuth tests, the router also needs a persistent high-entropy
`ROUTER_AUTH_KEK`, an HTTPS `ROUTER_PUBLIC_URL` reachable by the provider, and
`DASHBOARD_BASE_URL` set to the complete scoped Volt Connections URL, including
the organization/app path and config query. Keep the encryption key in
the ignored local environment and preserve it across restarts. Slack additionally
requires `SLACK_MCP_CLIENT_ID` and `SLACK_MCP_CLIENT_SECRET`; register
`<ROUTER_PUBLIC_URL>/v1/agents/connectors/oauth/callback` with the OAuth app.
Enable Slack MCP under the app's Agents settings and PKCE under OAuth settings.
Enabling MCP adds Slack's default scopes: remove scopes the test does not need
before requesting installation. For channel and DM sending, use `chat:write`
and `im:write`, plus `channels:read` for public-channel lookup and `users:read`
for user profiles. Workspace installation approval and the user's OAuth consent
are separate steps. Start consent from Volt so its callback state is present.
If using a temporary tunnel with the local noauth router, expose only OAuth
launch/callback and `/.well-known/oauth-client-metadata`, never its management
API. Verify the callback returns to the scoped Volt Connections page before
starting consent. Restart the router after changing these environment values.

## How Agents is wired

- Dev-only. `src/api/agents.ts` throws outside `import.meta.env.DEV`.
- Browser calls `/__agents/<orgId>/<appId>/v1/...`; `scripts/agents-dev-proxy.ts` (mounted in
  `vite.config.ts`) strips credentials, sets `X-Customer-Id` = app id, and forwards to
  `AGENTS_ROUTER_URL`. Configs under `examples` in the router won't show up; they live under
  the numeric app id.
- Requests go through `agentRequest` / `agentQueryOptions` (`src/queries/agents/`). Don't
  call `fetch` directly.
- Types: `src/types/agents.ts` re-exports from `src/gen/agents/` (generated). Never edit `src/gen/`.
- Routes: `src/routes/organization/$orgId/$appId/agents/`. Components:
  `src/components/custom/agents/`. Helpers: `src/utils/agents/`. Nav: `src/constants/sidebars.tsx`.

## Picking a model

Any field that picks an STT, TTS, LLM or other model uses the model picker, never a
`<select>` or free-text input. It renders a summary card: the label above, the current
choice in bold ("gemini / gemini-3.8-flash"), a muted detail line ("Pinned model.",
"The router's default.", "Now provider / model" for a route), and a **Change** button.
Change opens a dialog with two sections: **Routers** (route cards from
`/v1/<modality>/routes`, default first) and **Models** (a table of pinnable models from
`/v1/<modality>/providers`). The columns follow the modality: LLM shows popularity,
intelligence index, tokens per second and input and output price per million tokens, TTS
Elo and characters per second, STT WER and latency, search the search index and cost
per task (Artificial Analysis numbers from
`benchmark`, refreshed with the `refresh_model_stats` skill), anything else the live
popularity, latency, error rate, languages and speed.

- In a react-hook-form form: `AgentModelSelect` (`agent-model-select.tsx`) with `name`,
  `label`, `modality`, `purpose` (dialog description), `defaultRoute`, optional `hint`.
  It stores the bare target (a route id or `provider/model`); empty means the default route.
- When one choice spans several modalities (e.g. a text LLM or a realtime speech-to-speech
  model) or doesn't map to a single form field: `AgentModelPicker`
  (`agent-model-picker.tsx`) with `sources` and a `modality:target` value. See
  `agent-conversation-model.tsx`.

## Ordering a list of choices

`AgentModelChain` (`agent-model-chain.tsx`) is the reordering primitive: a form field
holding `string[]`, shown as one `AgentModelPicker` per entry in the order they are tried.
The first row takes the `label`, the rest are "Fallback 1", "Fallback 2". Each row has a
drag grip and, for keyboard and tests, **Move up** / **Move down** / **Remove** buttons
(`aria-label`s read "Move Fallback 1 up"). **Add fallback** appends `defaultRoute`.

Volt has no drag-and-drop dependency and this uses native HTML5 drag on the grip: the drag
image is set to the grip's row so the whole row follows the cursor, and `dragover` on a
neighbour reorders as it goes, so the list on screen is what dropping would leave behind.
Follow the same shape — grip plus move buttons — anywhere else rows are user-ordered,
rather than reaching for `dnd-kit`.

Use it for anything routing tries in order. A modality block's `providers` is a priority
list that wins over `target`, so write one entry back as `target` and several as
`providers`, clearing whichever is unused. Every type reads a list. LLM falls back down it
per failed response; search retries a failed search elsewhere only when a list is written,
a single target reports the failure instead.

## Changing the router API

`api/agents/openapi.yaml` is a copy of `acceleration/api/openapi.yaml`. After a spec change:
copy it over, run `bun run gen:agents`, and fix type errors. Route changes: `bun routes:generate`.

## Before calling it done

```bash
bun lint                 # must pass
bun run build
bun run vitest run tests/unit/agents
```

UI copy is sentence case; sidebar group titles are one word and labels don't repeat them
(see Volt's `CLAUDE.md`). Only use `<Icon>` from `components/ui/icon.tsx` in sidebars.
