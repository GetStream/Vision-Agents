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
`localhost`, or CORS/auth break). The override puts the router in `noauth` mode with Volt's
CORS origin. Rebuild the router after pulling router changes.

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
`/v1/<modality>/providers`, with popularity, latency, error rate, languages and speed).

- In a react-hook-form form: `AgentModelSelect` (`agent-model-select.tsx`) with `name`,
  `label`, `modality`, `purpose` (dialog description), `defaultRoute`, optional `hint`.
  It stores the bare target (a route id or `provider/model`); empty means the default route.
- When one choice spans several modalities (e.g. a text LLM or a realtime speech-to-speech
  model) or doesn't map to a single form field: `AgentModelPicker`
  (`agent-model-picker.tsx`) with `sources` and a `modality:target` value. See
  `agent-conversation-model.tsx`.

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
