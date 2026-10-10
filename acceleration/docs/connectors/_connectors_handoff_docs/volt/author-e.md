# Phase E author log (2026-10-09)

PR: https://github.com/GetStream/volt-dashboard/pull/998 (draft), branch connectors/agents-ui-e, head 1a6eca5299, base connectors/agents-ui 447393f091 (unchanged at push).
Worktree: $S/volt/wt-e. Scope: F47, F48 + Kanat's scope update F50, F51, F53 (F52, F55 router: untouched).

## Files
- dialogs/connection-create-dialog.tsx (new), widgets/connection-create-button.tsx (new)
- lib/connections.ts: CONNECTIONS_PATH exported, creatableSchemes, NewConnectionValues, newConnectionUser, connectionRequest, newConnectionSchema
- lib/use-connector-consent.ts: connect(connection, opened?) + openPopup returned
- connections route index.tsx (actions), widgets/connector-detail-page.tsx (button when a creatable scheme exists)
- dialogs/connector-binding-dialog.tsx (F48 `listing`, copy), widgets/connections-page.tsx (empty-state copy)
- widgets/connection-overview.tsx (F50), widgets/connection-invocations.tsx (F51), lib/connectors.ts revisionMoveNote + dialogs/connector-dialog.tsx (F53)
- src/test-ids.ts: connectionCreateButton, connectionCreateSubmit
- tests: tests/unit/agents/connection-create.test.tsx (new, fetch stub, 12 tests), agent-connectors.test.tsx (+3: F48 x2, copy), connections-page.test.tsx (stub answers /v1/agents/configs with [] — the F51 read)

## Checks
- bun lint 0; knip:check 0 (exports 317, ratchet improved); build 0; unit 236 files / 2691 tests pass ($S/volt/e-unit.log)
- tsc -p tsconfig.app.json: base 1117, head 1117, diff empty ($S/volt/e-tsc-base.txt, e-tsc-head.txt)

## Mutations ($S/volt/e-mut.py, log $S/volt/e-mutations.log) — all killed
- 0 F48 typed names while no listing: exit=1 failed=['connector bindings on the tools tab > names as many tools as needed while no connection lists them']
- 1 F48 listing gives checkboxes: exit=1 failed=['connector bindings on the tools tab > offers the tools a connection lists as checkboxes, with no names to type', 'connector bindings on the tools tab > saves a fixed binding with each granted tool pinned to its digest', 'connector bindings on the tools tab > saves a session binding that grants by name, with its policy']
- 2 F47 dialog copy: exit=1 failed=['connector bindings on the tools tab > points to where the app’s own connections are made']
- 3 F47 owner defaults to signed-in user: exit=1 failed=['New connection > creates one from a connector’s page, for that connector, with its inputs', 'New connection > opens the consent for an OAuth connection the signed-in user owns, by default', 'New connection > sends an API key and its header for an api_key connector']
- 4 F47 create acts as owner (header): exit=1 failed=['New connection > creates a connection for an end user named by ID, acting as them']
- 5 F47 end user by ID: exit=1 failed=['New connection > creates a connection for an end user named by ID, acting as them']
- 6 F47 token never in create body: exit=1 failed=['New connection > creates a connection for an end user named by ID, acting as them', 'New connection > creates an app token connection, then stores the token on it alone', 'New connection > creates one from a connector’s page, for that connector, with its inputs', 'New connection > opens the consent for an OAuth connection the signed-in user owns, by default', 'New connection > sends an API key and its header for an api_key connector']
- 7 F47 retry reuses created connection: exit=1 failed=['New connection > stores the token on the connection it made when the first try was refused']
- 8 F47 popup opened before request: exit=1 failed=['New connection > creates a connection for an end user named by ID, acting as them', 'New connection > opens the consent for an OAuth connection the signed-in user owns, by default']
- 9 F47 end user ID required: exit=1 failed=['New connection > creates a connection for an end user named by ID, acting as them']
- 10 F47 input without default required: exit=1 failed=['New connection > creates one from a connector’s page, for that connector, with its inputs']
- 11 F47 input pattern: exit=1 failed=['New connection > creates one from a connector’s page, for that connector, with its inputs']
- 12 F47 token/key required: exit=1 failed=['New connection > refuses a token connection with no token, before any request']
- 13 F47 list turns to new owner after consent: exit=1 failed=['New connection > opens the consent for an OAuth connection the signed-in user owns, by default']
- 14 F50 connected without account: exit=1 failed=['a connection’s detail page > shows no account for a connected one the provider names none for']
- 15 F51 session binding agent named: exit=1 failed=['a connection’s detail page > names the agent of a session binding’s call, as it names a fixed one’s']
- 16 F53 token connector copy: exit=1 failed=['a custom connector’s next revision > says a token connection moves when its token is saved again']

## Browser (:3011, own tab pageId 9; dev server PID 23083 stopped; certs/.env.local removed)
1. Connections tab: New connection dialog (connector picker, owner radio Signed-in user default). GitHub shows Authentication radio OAuth/Bearer token. Screenshot: dialog with GitHub, Bearer token, App, label ui-test-e-github-pat, Token masked.
2. Create → POST /connections 201 (reqid 2899) → PUT …/797fabfc475e645905a62bac09f9464e/credentials 200 (2901) → navigated to detail: Connected, Owner App, Account "—", Credential revision 2, audit Granted Access 3577528b (same fp as phase D: same PAT). Toast "Connection created".
3. Validate → POST validate 200 (2909), alert "Validation: Works", tools table listed (add_comment_to_pending_review …).
4. Delete via Actions › Delete, typed confirm → toast "Connection “ui-test-e-github-pat” deleted", back on list.
5. Linear OAuth as signed-in user: popup (page 10) at https://carol-elliptic-uncloak.ngrok-free.dev/v1/agents/connectors/oauth/launch/830a0df1… "Connecting your account / Connecting securely to the provider…". List moved to ?owner=user&connector_id=linear: ui-test-e-linear 79f5414a… Pending, Kanat's Linear 91d5e6f9 Connected (untouched). Popup closed (mine); ui-test-e-linear deleted (toast).
6. Linq page: New connection with connector preset (DescriptionList row, copy button removed), phone_number field, Token. Cancelled.
- Console: 2x 404 GET …/linq/oauth-client (none set), PostHog warn. No app errors.
- Secret: E2E_GITHUB_PAT filled by $S/volt/e-helper.py (copy of phase D helper, ALLOW = E2E_GITHUB_PAT only, ORIGIN :3011); helper killed, url file removed. Value never printed.
- Not done live: F48 in the binding dialog (needs an agent; covered by unit tests). Consent completion (only from :3000).

## State left
Nothing created remains: ui-test-e-github-pat and ui-test-e-linear deleted. Note: the Linear launch attempt 830a0df1… expires on its own (10 min).

## Questions / notes
- oauth2_client_credentials (Salesforce) and none have no create form; Salesforce shows only OAuth in the dialog. Add a client_id/client_secret form later?
- A refused token leaves the created connection pending if the dialog is then cancelled (retry in the same dialog reuses it). Acceptable, or delete on cancel?
