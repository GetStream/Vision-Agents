# Phase F author log (2026-10-09)

PR: https://github.com/GetStream/volt-dashboard/pull/999 (draft), branch connectors/agents-ui-f, head 466e491492, base connectors/agents-ui 329fc17161 (unchanged at push).
Worktree: $S/volt/wt-f. Router spec: Vision-Agents origin/accelerate 599298c6.

## Files
- api/agents/openapi.yaml + src/gen/agents (gen:agents): redirect_uri on Connector, deleteConnector. Nothing broke.
- dialogs/connector-delete-dialog.tsx (new): unforced DELETE; on 409 setInUse, rethrow (router message, or "Connections or agent configs use this connector." when agentFailure dropped it, code !== 'conflict'); second confirm "Delete anyway" sends ?force=true. Invalidates CONNECTOR_LISTS + CONNECTION_READS.
- widgets/connector-detail-page.tsx: ActionsMenu › Delete on custom only (disabledReason for viewers); navigate to catalog, then removeQueries of connectorPath/oauthClientPath.
- lib/connectors.ts: CONNECTOR_LISTS ('/v1/agents/connectors?'), takesRedirectUri, NO_REDIRECT_URI.
- sections/connector-oauth-client-section.tsx: Redirect URI DescriptionList row (copyable) on oauth2_code; "None: ..." when absent.
- dialogs/connector-oauth-client-dialog.tsx: ReadOnlyField + Chip onCopy (pattern of agent-app-setup.tsx:236).
- tests/unit/agents/connectors-page.test.tsx: stub answers DELETE /connectors/{id} (404 built-in, 409 from `uses` map unless force=true, 204); +9 tests.
- Step 4: copy kept. #870 last commit "keep a custom connector's endpoint hidden (AI-837)"; Connector schema has no endpoint.

## Checks
- bun lint 0 (f-lint.log); knip:check 0 (f-knip.log); build 0 (f-build.log); unit 236 files / 2715 tests pass (f-unit.log, run with .env.local copied in, removed after; without it 157 env-only failures VITE_AMPERE_API).
- tsc -p tsconfig.app.json: base 1121, head 1121, diff empty (f-tsc-base.txt, f-tsc-head.txt).

## Mutations ($S/volt/f-mut.py, log $S/volt/f-mutations.log): 23/23 killed
- 0 first try unforced: exit=1 failed=['Deleting a custom connector > deletes one nothing uses without force, then returns to the catalog', 'Deleting a custom connector > names what uses one, and deletes it with force only on a second confirm', 'Deleting a custom connector > refetches the connections a forced delete deleted', 'Deleting a custom connector > says connections or agent configs use one when the router’s list is too long to show']
- 1 force offered only after a 409: exit=1 failed=['Deleting a custom connector > names what uses one, and deletes it with force only on a second confirm', 'Deleting a custom connector > refetches the connections a forced delete deleted', 'Deleting a custom connector > says connections or agent configs use one when the router’s list is too long to show']
- 2 force sent as force=true: exit=1 failed=['Deleting a custom connector > names what uses one, and deletes it with force only on a second confirm', 'Deleting a custom connector > refetches the connections a forced delete deleted']
- 3 no second force without 409 (inUse set): exit=1 failed=['Deleting a custom connector > names what uses one, and deletes it with force only on a second confirm', 'Deleting a custom connector > refetches the connections a forced delete deleted', 'Deleting a custom connector > says connections or agent configs use one when the router’s list is too long to show']
- 4 router message shown on 409: exit=1 failed=['Deleting a custom connector > names what uses one, and deletes it with force only on a second confirm']
- 5 fallback when router list dropped: exit=1 failed=['Deleting a custom connector > says connections or agent configs use one when the router’s list is too long to show']
- 6 forced copy says bindings stay: exit=1 failed=['Deleting a custom connector > names what uses one, and deletes it with force only on a second confirm']
- 7 invalidate connections: exit=1 failed=['Deleting a custom connector > refetches the connections a forced delete deleted']
- 8 invalidate catalog lists: exit=1 failed=['Deleting a custom connector > deletes one nothing uses without force, then returns to the catalog']
- 9 no refetch of the deleted connector: exit=1 failed=['Deleting a custom connector > deletes one nothing uses without force, then returns to the catalog']
- 10 deleted connector reads removed: exit=1 failed=['Deleting a custom connector > deletes one nothing uses without force, then returns to the catalog']
- 11 typed id confirmation: exit=1 failed=['Deleting a custom connector > deletes one nothing uses without force, then returns to the catalog', 'Deleting a custom connector > names what uses one, and deletes it with force only on a second confirm', 'Deleting a custom connector > refetches the connections a forced delete deleted', 'Deleting a custom connector > says connections or agent configs use one when the router’s list is too long to show']
- 12 return to catalog: exit=1 failed=['Deleting a custom connector > deletes one nothing uses without force, then returns to the catalog', 'Deleting a custom connector > names what uses one, and deletes it with force only on a second confirm', 'Deleting a custom connector > refetches the connections a forced delete deleted']
- 13 delete only on custom: exit=1 failed=['Deleting a custom connector > offers no delete on a built-in connector']
- 14 delete disabled for viewer: exit=1 failed=['Deleting a custom connector > lets a viewer who cannot manage the app not delete one']
- 15 section shows redirect row: exit=1 failed=['Connector page > says there is no redirect URI when the router leaves it out', 'Connector page > shows the redirect URI an OAuth connector’s client has to list, to copy']
- 16 section copy: exit=1 failed=['Connector page > shows the redirect URI an OAuth connector’s client has to list, to copy']
- 17 section absent copy hidden: exit=1 failed=['Connector page > says there is no redirect URI when the router leaves it out']
- 18 section absent message: exit=1 failed=['Connector page > says there is no redirect URI when the router leaves it out']
- 19 dialog shows redirect: exit=1 failed=['Connector page > says there is no redirect URI when the router leaves it out', 'Connector page > shows the redirect URI an OAuth connector’s client has to list, to copy']
- 20 dialog copies uri: exit=1 failed=['Connector page > shows the redirect URI an OAuth connector’s client has to list, to copy']
- 21 dialog absent message: exit=1 failed=['Connector page > says there is no redirect URI when the router leaves it out']
- 22 only on oauth2_code: exit=1 failed=['Connector page > shows no redirect URI on a connector that connects without OAuth']

## Browser (:3011, own tab pageId 14; dev server PID 40635 stopped; certs/.env.local removed)
1. linear page: OAuth client section "Redirect URI https://carol-elliptic-uncloak.ngrok-free.dev/v1/agents/connectors/oauth/callback" + Copy; no Actions menu (built-in). Screenshot taken.
2. Created custom_uitest_f_bearer (ui-test-f-bearer, bearer) from Add custom connector: POST 200 (reqid 5758). Page: no Redirect URI row (bearer).
3. Agent ui-test-f-agent (303e8305b48264c5ddeaa2490d9f23fd) via proxy POST /v1/agents/configs 201 (5763), session binding 'bearer' → custom_uitest_f_bearer.
4. Actions › Delete, typed id: DELETE 409 (5764); banner "Custom_uitest_f_bearer is used by agent config bindings "ui-test-f-agent" as bearer: delete or unbind them first, or delete with force=true."; body adds forced copy; button "Delete anyway". Screenshot taken.
5. Delete anyway: DELETE ?force=true 204 (5765) → catalog. First build then refetched GET connector 404 (5766) + oauth-client 404 (5767) before leaving → fixed (CONNECTOR_LISTS + removeQueries after navigate), added test + mutations 9/10.
6. After fix: custom_uitest_github (phase D leftover, 0 connections) DELETE 204 (14550) → catalog, next GET only connectors?limit=25 (14551), no read of the deleted connector.
7. Agent read after forced delete: binding still names custom_uitest_f_bearer (AI-1051 ok); agent DELETE 204. Custom connectors left: none; ui-test configs left: none.
8. HubSpot Set OAuth client dialog: Redirect URI chip with copy; cancelled.
- Console: PostHog warn, 404 on oauth-client reads (none stored, pre-existing). No app errors.

## Notes / questions
- The router's 409 text is passed through agentFailure's sentence(): the id is capitalized ("Custom_uitest_f_bearer") and it ends in "or delete with force=true" (API wording in a UI). Changing either means touching shared utils/agents/errors.ts or the router message.
