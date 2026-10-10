# Reviewer F, round 1: volt-dashboard PR #999 (2026-10-09)

VERDICT: NO-GO (1 Should fix: 6 surviving mutations)

REVIEWED: head 466e491492 merged with origin/connectors/agents-ui 329fc17161 ("Already up to date"). Router: Vision-Agents 599298c6 (= origin/accelerate head at review time).
Worktrees: $S/volt/wt-f-rev (head), $S/volt/wt-f-rev-base (base, tsc only, node_modules symlinked). Both removed at the end.
SCOPE: the 8 touched files. Callees read: use-agents-mutation.ts, entity-delete-dialog.tsx, api/agents.ts (AgentRequestError), utils/agents/errors.ts (agentFailure, readable: >280 chars or {}<> drops the router text), lib/connections.ts (CONNECTION_READS, CONNECTORS_PATH), agent-query-options.ts, route $connectorId.tsx loader. Router at 599298c6: api/connectors.go deleteConnector:348, connectorInUse:376, connectorOf:462; store/connectors.go DeleteConnectorDefinition:481-575; api/connections.go connectionDeleted:492; apierror.go conflict code "conflict".

## Checks (merged tree)
- install 0, lint 0 (tree clean after --fix), knip:check 0 (exports 317), build 0. Logs $S/volt/rvf.{install,lint,knip,build}.log
- unit: 236 files, 2715 tests pass ($S/volt/rvf.unit.log; .env.local copied in, removed with the worktree).
- tsc -p tsconfig.app.json: base 1121, head 1121, sorted diff empty, 0 errors in touched files ($S/volt/rvf.tsc.{base,head}.sorted).
- Spec: `git show 599298c6:acceleration/api/openapi.yaml | diff - api/agents/openapi.yaml` → identical. `bun run gen:agents` exit 0, git status clean ($S/volt/rvf.gen.log).

## Router match
- Delete only on custom: ActionsMenu behind `connector.custom` (connector-detail-page.tsx:75); live: linear has no Actions menu, custom one does. Router 404s a built-in (errNoCustomConnector).
- First try unforced: mutateAsync(inUse) with inUse=false (dialog:51). Force only after a 409 (dialog:53-63), sent as ?force=true (dialog:31), which Huma reads as bool.
- 409 copy: router message passed through when agentFailure kept it (code 'conflict'), else "Connections or agent configs use this connector."; forced consequence says connections go, credentials drop, providers not asked to revoke, bindings stay (AI-1051). Matches store doc (softDeleteConnection; bindings left) and connectionDeleted (no provider revocation). Scope text (OAuth client, config token, event destinations) matches store:570.
- Cache: invalidates CONNECTOR_LISTS ('/v1/agents/connectors?', covers catalog and CONNECTORS_PATH ?limit=200) + CONNECTION_READS; the detail and oauth-client reads are removed after navigate, not refetched.
- Redirect URI: router sets it only when schemes contain oauth2_code (connectorOf:483); UI gates on the same (lib/connectors.ts takesRedirectUri). Absent → NO_REDIRECT_URI text, no copy.

## Browser (:3014, own tab 15 closed; dev PID 45635 stopped; other tabs 3/9/14 untouched)
1. linear: OAuth client section "Redirect URI https://carol-elliptic-uncloak.ngrok-free.dev/v1/agents/connectors/oauth/callback" + Copy; no Actions menu. Screenshot taken.
2. Add custom connector custom_uitest_rvf (ui-test-rvf-bearer, bearer): POST 200 (reqid 5752). Page: "Custom" tag, Actions menu, no Redirect URI row.
3. Actions › Delete: body "…and every revision of it, with its OAuth client, config token and event destinations. If connections or agent configs use it, they are named before anything is deleted. It can't be undone."; typed id; DELETE (unforced) 204 (5757) → catalog, toast "Connector “ui-test-rvf-bearer” deleted". After the DELETE only GET connectors?limit=25 (5758, 5759); no read of the deleted id.
- Console: 2x 404 = oauth-client reads before the delete (5755/5756, pre-existing: none stored). Nothing left behind.

## Findings
F-1 [Should fix] tests/unit/agents/connectors-page.test.tsx — 6 of my 11 mutations survive:
  R23 dialog:43 scope text cut to "and every revision of it" (what goes: OAuth client, config token, event destinations);
  R24 dialog:46 "and drops their credentials at once" removed; R25 dialog:46 "The providers are not asked to revoke…" removed (the test matches only /Deleting it anyway deletes its connections/ and the bindings sentence);
  R31 dialog:47 unforced consequence removed;
  R30 dialog:65 `throw error` → `return`: a non-409 failure (404, 500) then closes the dialog with a success toast and navigates away; no test sends a failing delete;
  R28 connector-detail-page.tsx:154 oauthClientPath dropped from the removal: the stale oauth-client read stays cached for a re-created id (router allows the same id again).
  Fix: assert the full dialog body before and after the 409; add a test where DELETE answers 500 (or 404) → dialog stays open with the error, no navigation, no toast; assert the oauth-client read is refetched on coming back (or drop that removal).
F-2 [Nit] dialogs/connector-oauth-client-dialog.tsx:33 docstring line is 136 chars, not rewrapped (same as phase E F-5).
F-3 [Nit] sections/connector-oauth-client-section.tsx: the Redirect URI shows on connectors that take no own client (linear: "Registered at each consent"), where nobody registers a client by hand. Matches the router contract; consider hiding it when !takesOwnClient. Ticket.
F-4 [Nit] dialog:43 "config token" is router vocabulary; nothing else in the dashboard names it. Ticket if copy is revisited.
Author QUESTION (409 casing/"force=true"): accepted by orchestrator, ticketed; not counted.

## Mutations
Author sample re-run (0 2 3 5 7 9 10 13 15 22): 10/10 killed. Mine R23-R33: 11 checked, 5 killed (R26 navigate target, R29 confirm label, R32/R33 oauth2_code gate in dialog and section), 6 survived (F-1), R27 (removeQueries before navigate) survived but is equivalent: removeQueries does not notify the mounted observer, so no refetch.
Script $S/volt/rvf-mut.py (author's 0-22 + mine 23-33; `python3 rvf-mut.py <idx…>`, edit W to the current worktree), log $S/volt/rvf-mutations.log.

## Design standards
dashboard/ tier only (dialogs/, widgets/, sections/, lib/): ok. Primitives: ActionsMenu (shared/widgets, same spot as agent-page.tsx:158, connection-detail-page.tsx:118), EntityDeleteDialog, DescriptionList, ReadOnlyField + Chip onCopy (same as agent-app-setup.tsx:236). No Remixicon, no custom/ import, no raw fetch. Copy is sentence case. disabledReason MANAGE_AGENTS_MESSAGE for viewers. No new test ids needed.

## UNVERIFIED
- 409 → force path live: the author did it (reqs 5764/5765); I did not, since the brief said delete without binding.
- The absent-redirect_uri message live (local router has ROUTER_PUBLIC_URL set); unit test only.
