# Phase A author log (2026-10-09)

PR: https://github.com/GetStream/volt-dashboard/pull/994 (draft), head `connectors/agents-ui-a` 5354652b39, base `connectors/agents-ui` c3e9244b7e (= origin/ai-team/agent-dashboard).
Worktree: $S/volt/wt-a. Old names `kanat/agents-connectors` were pushed first, then deleted on the remote (`git push origin --delete kanat/agents-connectors`). `-a` under the old name was never pushed.

## Files
- api/agents/openapi.yaml: copied from Vision-Agents origin/accelerate c000cedc. src/gen/agents/endpoints/agents.d.ts was produced with `bun run gen:agents`.
- src/utils/agents/connector-consent.ts: launchOrigin, startConsent, consentOutcomeFrom, finishConsentLanding.
- src/main.tsx: calls finishConsentLanding() before render, the same way plugin_connected is handled.
- src/components/dashboard/agents/lib/connections.ts: paths, labels, query options keyed by user, argumentShapes, credentialLines, credentialValues.
- lib/use-connector-consent.ts: opens the popup in the click, POSTs …/authorizations as the owner, toasts the outcome, invalidates connection reads.
- widgets/connections-page.tsx, widgets/connection-detail-page.tsx, dialogs/connection-credentials-dialog.tsx, dialogs/connection-delete-dialog.tsx.
- routes agents/resources/connections/{index,$connectionId}.tsx; sidebar group Resources > Connections (icon `link`); test-ids AgentsResourcesSelectors.
- src/api/agents.ts: `agentRequest(..., { userId })` overrides X-Stream-User-Id. The actor stays the dashboard user. useAgentsMutation takes `userId`.
- tests/unit/agents/connector-consent.test.ts (16), tests/unit/agents/connections-page.test.tsx (14).

## Design decisions
- The landing runs in main.tsx, not as a route. The router redirects to DASHBOARD_BASE_URL exactly, with no path (authorizations.go:643-655). The same variable also builds the plugin redirect (plugins/oauth.go:283-289), so it cannot carry a dedicated path.
- The opener is notified on a BroadcastChannel (`va.connector.oauth`), not on window.opener. The launch page sets `window.opener = null` before it goes to the provider (authorizations.go launchPage script).
- A ready message is accepted only when event.source === popup and event.origin === the launch_url origin. The handoff postMessage uses that origin as targetOrigin. The launch_url must be https; http is accepted only for localhost.
- The page lists one owner at a time because the API has no "all users" listing: owner_type=user needs X-Stream-User-Id. The default user is the signed-in `volt-<id>`, the same id the playground sessions use.
- Delete sends `?force=true` only when `used_by` is non-empty, and the dialog names the configs that bind the connection.
- The replace dialog is shown for bearer (token) and api_key (header + api_key). Reconnect/Connect is shown for oauth2_code. Other schemes get Validate and Delete only.

## Evidence
- `bun lint`: pass. `bun run build`: exit 0. `bun run knip:check`: pass (exports 316 ≤ baseline 317).
- `bun run vitest run --project unit`: 231 files, 2612 tests pass.
- Mutation checks: each of these, when removed, makes exactly 1 test fail: `event.origin !== origin`, `event.source !== popup`, targetOrigin → '*', `data.connection_id !== connectionId`.
- Dev server on https://local.getstream.io:3011 (PORT=3011 BROWSER=none vite). Requests went through /__agents/1181507/1257545:
  - GET /v1/agents/connections?owner_type=app&limit=25 → 200, 0 items (the app has none)
  - GET /v1/agents/connectors?limit=200 → 200, 18 items
  - A temporary connection was created with POST /v1/agents/connections (github, bearer, label volt-phase-a-check, id 331ba4f970fb922517eff739871c6649). With it in place:
    - the list returned it (pending/current)
    - tools, invocations and connector-audit → 200
    - validate → 200 status=pending
    - DELETE → 204, after which the list was empty again
  - Vite served the transformed modules for both widgets, the route and the utility with 200.
- The dev server was stopped (port 3011 is free). The copied certs were removed from the worktree.

## Gaps / questions
1. No signed-in browser render. The chrome-devtools browser has no dashboard session and redirected to /login/, and I did not log in. Phase D should load the page on :3000.
2. The consent E2E only works from the origin in DASHBOARD_BASE_URL (http://local.getstream.io:3000). The launch page checks event.origin === that origin, so a dev server on :3011 cannot finish it. Also, ROUTER_PUBLIC_URL is the ngrok URL, so launch_url is https on ngrok, which is fine.
3. The API has no `connected_at` field on Connection (the DB has the column). The page shows Created and Updated instead. A router change could expose it.
4. mkcert: with a fresh certs dir, vite-plugin-mkcert tries `mkcert -install`, which needs sudo, and fails. A worktree needs the main checkout's certs dir copied in.
5. Anyone same-origin can send the landing broadcast, for example by opening a crafted `/?connection_id=X&status=connected` link. The opener only toasts and refetches, and the real state comes from the refetch.
