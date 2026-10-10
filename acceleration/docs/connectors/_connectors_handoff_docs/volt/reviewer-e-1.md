# Reviewer E, round 1: volt-dashboard PR #998 (2026-10-09)

VERDICT: NO-GO (2 Should fix: orphan pending rows, surviving mutations)

REVIEWED: head 1a6eca5299 merged with origin/connectors/agents-ui 447393f091 ("Already up to date": base did not move). Router: Vision-Agents origin/accelerate 26051062.
Worktrees: $S/volt/wt-rv-e (head), $S/volt/wt-rv-e-base (base, tsc only). Both removed at the end.
SCOPE: the 16 touched files. Callees read: use-agents-mutation.ts, api/agents.ts (agentRequest, agentUser), utils/agents/connector-consent.ts. Router: connections.go (createConnection, ownerOf, mayReach, defaultScheme), connection_tools.go (putConnectionCredentials), schemes/bearer and apikey Complete, openapi ConnectionRequest and ConnectionCredentials.

## Checks (merged tree)
- install 0, lint 0, knip:check 0 (exports 317), build 0. Logs: $S/volt/rve.{install,lint,knip,build}.log
- unit: 236 files, 2691 tests pass ($S/volt/rve.unit2.log). The first run had 157 failures because .env.local was missing (VITE_AMPERE_API), not because of the PR ($S/volt/rve.unit.log).
- tsc -p tsconfig.app.json: base 1121, head 1121, and the sorted diff is empty ($S/volt/rve.tsc.{base,head}.sorted). The author counted 1117 on each side. The difference comes from the environment, and the delta is 0 in both runs.

## Router match (F47)
- POST /v1/agents/connections. The body is {connector_id, auth_scheme, owner, label?, inputs?} and has no token (live req 2899: `{"connector_id":"github","auth_scheme":"bearer","owner":{"type":"app"},"label":"ui-test-rv-e-refused"}`). This matches ConnectionRequest (additionalProperties false; connections.go:146-151). An app owner sends no user_id (ownerOf connections.go:578). A user owner sends owner.user_id and the same X-Stream-User-Id (dialog:71 `userId: (body) => body.owner.user_id`). That satisfies ownerOf:589 (acting == userID). The live Linear create as volt-1115938 returned 201 (req 5809) and the list shows owner volt-1115938.
- PUT .../credentials {expected_revision: connection.revision, values: credentialValues(...)}. Bearer sends {token}, api_key sends {api_key, header}. This matches ConnectionCredentials and bearer/apikey Complete. The user header comes from connectionUser (mayReach connections.go:566).
- OAuth: the popup opens inside the submit before any request. consent.connect(connection, popup) sends POST .../authorizations (req 5811 201) and the popup goes to the ngrok launch page (phase A handoff).
- The token is never echoed or stored. In-page check after the create: DOM, inputs, localStorage, sessionStorage and the URL do not contain the PAT (all false). The router log (10 min) and the dev log have 0 hits for the PAT value (grep -cF). The refusal message names no value: "The credentials were refused: bearer: the token is not an RFC 6750 b64token…". Audit and log record only fp 3577528b (grant_created 22:28:07Z).

## Browser (:3014, my own tab 11 and its popup 12, both closed; dev PID 31340 stopped; helper PID 31825 stopped, url file removed)
1. GitHub, Bearer, App, label ui-test-rv-e-refused, token "not a token!". POST 201 0fae1102…, PUT 400, alert shown inline. Cancel. **The list shows ui-test-rv-e-refused, Pending** (orphan, F-1). I deleted it (DELETE 204, req 2902).
2. GitHub PAT, App, ui-test-rv-e-github-pat. Create took me to the detail page: Connected, Account "—" (F50 ok), credential revision 2, audit Granted Access 3577528b, toast "Connection created". Deleted (typed confirm), back on the list.
3. Linear: New connection, then Cancel. No request was made, and the list (?owner=user&connector_id=linear) holds only Kanat's 91d5e6f9 Connected. **No pending row.**
4. Linear OAuth submit: popup at the ngrok launch page and the list at ?owner=user&connector_id=linear. I closed the popup without consenting. **ui-test-rv-e-linear 60b62bdf… stays Pending** and no toast appears (F-3). I deleted it (DELETE 204, req 5813).
- Console: only the expected 400 from step 1, a PostHog warning and the password-field verbose notes.
- F48 was not tried live: the app has no agent config ("New agent" only), and I did not create one.

## Findings
F-1 [Should fix] src/components/dashboard/agents/dialogs/connection-create-dialog.tsx:62-64,106 and widgets/connection-create-button.tsx:54. A connection kept after refused credentials (`made`) is never deleted. Cancel (onDismiss only does setOpen(false)) leaves it Pending (live, step 1). A retry after changing the connector, scheme or owner creates a second connection and orphans the first (dialog:116-122). A retry with a changed label or inputs reuses the old connection and silently drops the edit (the reuse check ignores label and inputs). The orchestrator decided that cancel must delete. Fix: on dismiss, and whenever `made` is not reused, DELETE `made` (with connectionUser header) if it is still pending; include label and inputs in the reuse check, or recreate. Add tests for cancel and for an owner change after a refusal.
F-2 [Should fix] tests/unit/agents/connection-create.test.tsx, agent-connectors.test.tsx: 14 of my 20 mutations survive (list below). Add a test per rule, or delete the code.
F-3 [Question → ticket] An abandoned OAuth consent (popup closed) leaves a Pending connection made by the dialog (live, step 4). The dialog has closed by then, so the cancel rule does not cover it literally. The orchestrator decides whether it is in scope.
F-4 [Nit] connection-create-dialog.tsx:153 NewConnectionFields is a second component in the file (.claude/rules/code-style.md:5 "One component per file"). Base has the same shape in connector-binding-dialog.tsx:115 (BindingFieldsBody), and AGENTS.md:218 says a dialog includes its form.
F-5 [Nit] dialogs/connector-dialog.tsx:29 has a 137-char docstring line that was not rewrapped.

## Mutations
Author's sample re-run (0 1 4 6 7 8 13 14 15 16): 10 of 10 killed ($S/volt/rve-mut-author.log, script $S/volt/rve-mut-author.py).
Mine ($S/volt/rve-mut.py, log $S/volt/rve-mutations.log; run `python3 rve-mut.py [idx…]`): 20 checked, 4 killed (R10 label omitted when empty, R13 mixed-scheme note, R14 pending copy, R17 api_key header required), 16 survived:
- Real untested rules (F-2): R1 reuse ignores the owner change, R2 reuse ignores the scheme change, R3 popup not closed when create fails, R4 F48 chips while tools load, R5 F48 checkbox group when a connection lists 0 tools (the "none listed, type names" rule), R6 inputs kept across a connector switch (stale inputs go to the next connector, and the router refuses unknown inputs), R7 connectors with no creatable scheme offered in the picker, R8 input enum check, R9 empty inputs sent, R12 New connection button on a connector with no creatable scheme, R15 OAuth-first default scheme order (matches router defaultScheme), R16 dialog stays open on consent, R20 connector-page consent navigates away.
- Equivalent or not new (not counted): R11 user_id trim (the zod schema already trims), R18 "Not listed" note (shown only when source is set, as before), R19 expected_revision 1 (a new connection is always revision 1).

## Nits F50 / F51 / F53
- F50: correct. Account shows "—" when connected without an account id and "Not connected yet" when pending (overview:81-84; live step 2).
- F51: correct. Agent names come from GET /v1/agents/configs, and used_by overrides them (invocations:43-53). Killed by author mutation 15.
- F53: correct. revisionMoveNote copy follows putConnectionCredentials (connection_tools.go:165-168). All three branches are tested (R13 killed, author 16 killed).
- F48: logic correct (`listing = !!source && (tools.isPending || listed.length > 0)`); the ChipInput is kept after the first chip (author 0 killed). R4 and R5 are untested.
- Dialog copy "Library › Connectors › Connections" matches sidebars.tsx:845/866 and connectors-layout tabs.

## Design standards
dashboard/ tier only (dialogs/, widgets/, lib/): ok. Primitives: Button, Radio, DescriptionList, Text, Form.*, FormDialog from design-system: ok. No Remixicon, no custom/ import, no raw fetch (diff grep). PrivilegeTooltipGuard on the button: ok. Test ids are BEM in src/test-ids.ts:1139-1140: ok. Copy is sentence case, and "pop-up" matches the hook's "Allow pop-ups". Labels sit above the fields. One component per file: F-4.

## UNVERIFIED
- F48 in a real browser (no agent config exists; unit tests and mutations only).
- A completed consent from a New connection (I did not authorize Linear) and whether the router expires pending rows from abandoned launches.
- The request body of the live OAuth create (req 5809): the MCP could not save it to the scratchpad. The owner was confirmed from the list instead.
