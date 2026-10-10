# Reviewer G, round 1 — volt-dashboard PR #1000 (2026-10-09)

VERDICT: NO-GO (5 Should fix: surviving mutations; the code itself behaves correctly in every case I probed)
REVIEWED: db69ed017d merged onto connectors/agents-ui 2a5e3f0a19 ("Already up to date"). Router facts at Vision-Agents 599298c6.
Worktrees $S/volt/wt-rg and wt-rg-base (base tsc). Both removed. .env.local and certs removed. Dev server PID 63987 (:3014) stopped. :3000 PID 48732/48734 not touched.

## Checks (merged tree)
- bun lint 0 (rg-lint.log). knip:check 0 (rg-knip.log). build 0 (rg-build.log).
- tsc -p tsconfig.app.json: base 1124, head 1124. The diff of sorted errors, positions stripped, is empty (rg-tsc-base.txt, rg-tsc-head.txt).
- Unit suite with .env.local: 236 files, 2739 tests, all pass (rg-unit.log).

## Router checks (599298c6)
- F60, validate shape: `ConnectionValidation{connection_id, status, code?, missing_scopes?, error?, tools_digest?, checked_at?}` (api/connection_tools.go:101-109). `validationAfter` (:438-482) returns needs_reauthorization for a 401 that `Resolver.Invalidate` moved (resolver.go:199-238). It returns `failed` for any other cause. `validationFailed` (connections.ts:152) treats every status except connected and needs_scopes as a failure. That is correct.
- F60, retry revision: `Invalidate` changes only the status. pgsealed AGENTS.md:47 says "anything else keeps the revision". So the retry PUT on `stored.revision` is right, also after a 401.
- F63: a fixed binding must name an app-owned connection (configs.go:562). So the app-scoped read with no user_id is the right one. `isNotFound` is true only for an AgentRequestError with status 404 (api/agents.ts:122). A network error or a 500 does not mark the binding Broken. During loading, error and catalog are both undefined, so nothing is marked Broken. The catalog read is `?limit=200` with a `has_more` guard. listConnectors and unboundConnectors both read the same store (connectors.go:270-304, configs.go:542).
- F62: the provider manifests list registration customer, managed, operator, dcr or cimd. slack.yaml has `[operator]` only, so `operator` in keepsOAuthClient is load-bearing.

## Mutations ($S/volt/rg-mut.py; log rg-mutations.log): 17 checked, 5 survived
Author sample 0, 4, 7, 9, 14, 15, 17, 24: all 8 killed.
Mine:
- R26 `isNotFound(...)` -> `!!fixedRead(binding)?.error`: SURVIVED. No test checks that a 500 or network error leaves the binding unbroken.
- R27 a catalog still loading is read as empty: killed.
- R28 `validationFailed` -> `status === 'failed'`: SURVIVED. No create test covers a 401 (needs_reauthorization), which is the path most providers take.
- R29 removeQueries without `read === path`: killed.
- R30 list row without the revision guard: SURVIVED. The revision-guard test covers only the detail page.
- R31 keepsOAuthClient without operator: SURVIVED. No test covers an operator-only connector such as slack.
- R32 used_by guard removed: killed (crash).
- R33 provider reason dropped from the message: killed.
- R34 the F68 reads run while the dialog is closed: SURVIVED.

## Browser (:3014, own tab 21, closed)
1. New connection › GitHub › Bearer › App, label ui-test-rg-bad-token, wrong token. Requests: POST 201 (5754), PUT 200 (5756), POST validate 200 (5758), GET connection 200 (5760). Inline alert: «The token did not work with the provider: mcp: connect to github: calling "initialize": sending "initialize": Bad Request. Enter another one, or cancel to delete the connection.» Screenshot taken. Cancel -> DELETE 204 (5761) -> list refetch.
2. Connection ui-test-rg-conn ede5fb96 (POST 201, PUT 200 via page fetch) and agent ui-test-rg-agent 8ce4813c (POST configs 201, fixed binding). Tools tab: «App connection ui-test-rg-conn · 0 tools» (F64 OK). Agent delete dialog: «…No other agent config binds ui-test-rg-conn, so messages that arrive on it, such as a Slack bot’s, go unanswered.» Cancelled.
3. DELETE connection ?force=true 204 (11549). Client-side nav Behavior -> Tools after 6 s: GET connection 404 (11583). Row «GitHub · github / Broken / Its connection ede5fb96… no longer exists, so sessions open without it. Choose another connection or remove it to save the agent. / Choose connection / Remove».
4. Cleanup: DELETE config 204. Remaining configs: [ui-test-slackbot-agent]. Remaining app connections: [ui-test-slackbot]. Kanat's items not touched.
Console: PostHog warn; 403s are non-agents resources (no agents fetch returned 403); 400 voices/library (env, known); 404 = the deleted connection read (expected).

## Honesty without the router changes
- "Last check" is only in the query cache. The revision guard drops a stale answer. After a reload the page shows Connected and no Last check line, which is the same claim as base: nothing false is added. OK; persisting the result is a router ticket.
- F68 copy shows for any fixed connection that only this config binds. For a GitHub connection the clause «messages that arrive on it» describes something that never happens. This is a Nit; the channel flag is a router ticket.

## Design standards
dashboard/ tier only. No Remixicon, no raw fetch, no custom/ import (grep of the added lines). Uses the design-system Tag (color red, the same as connection-invocations.tsx:96) and the existing TableCellStack. Sentence case. No new test ids needed. The PR body has the checklist. OK.
