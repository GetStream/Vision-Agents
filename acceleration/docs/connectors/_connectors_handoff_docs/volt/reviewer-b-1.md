# Reviewer B, round 1 — volt PR #996 @ 0096df3f42 (base connectors/agents-ui 744c52f775)
VERDICT: NO-GO (5 Should fix)

## Setup
- Worktree $S/volt/rv-b (detached 0096df3f42). Base 744c52f775 is an ancestor of the head, so the merge is a no-op. A base tree $S/volt/rv-b-base shared rv-b's node_modules through a symlink, because bun.lock is unchanged. Both trees are removed.
- Router source: Vision-Agents origin/accelerate 2605106222 (spec copied to $S/volt/rvb-openapi.yaml), plus internal/api/connectors.go, oauth_clients.go and connector_provider_apps.go.
- /pr-review was not run. That skill reviews Vision-Agents PRs (OpenAPI spec, SDK order). This PR is in volt-dashboard, so the review was done by hand against the brief's checklist.

## Checks (logs $S/volt/rvb.*.log)
- bun install 0, bun lint 0 (oxlint --fix left the tree clean), knip:check 0, build 0, vitest unit 232 files / 2631 tests pass.
- `npx tsc -p tsconfig.app.json`: 1121 errors on head and 1121 on base. The diff of the sorted error lines is empty ($S/volt/rvb{,-base}.tsc.sorted).
- Generated types (src/gen/agents/endpoints/agents.d.ts): StoredConnectorOAuthClient, ConnectorOAuthClientRequest, CustomConnectorRequest and ConnectorClient match the Go structs at 26051062.

## Router API match
- GET /connectors?limit&q&cursor matches listConnectorsRequest (q ≤120, limit ≤200). GET /{id} and POST match too.
- oauth-client GET/PUT/DELETE: paths, body fields and the 404-means-none rule are correct. Only a `customer` record is editable: the router answers 409 to a PUT over a managed or operator record (oauth_clients.go:254), and DELETE removes only a customer record (:315).
- provider_app_id and signing_secret ARE fields of ConnectorOAuthClientRequest (oauth_clients.go:96-102, a customer-registered app). PUT /provider-app is a different operation: it is the **managed** flow, where the router creates the Slack app from {name, config_refresh_token, allowed_ip_address_ranges} (connector_provider_apps.go:59-70). The PR is right to send those fields to oauth-client. The coordinator's premise was wrong. Managed creation is not in phase B's plan (plan.md:7), so it is not a finding.
- Registration rules: takesOwnClient = registration includes customer, which matches checkOAuthClient:353. The custom form refuses oauth2_code without a registration (connectors.go:300) and does not offer operator (:288). The schema's provider-app-only rule mirrors oauth_clients.go:359-363, except the router's "not when the connector lists oauth2_code" part, which the router reports as a 400.

## Findings
1. [Should fix] Untested new rules (surviving mutations; the list is below). The oauthClientSchema superRefine at lib/connectors.ts:89-113 has 4 rules and none is tested: R1, R2, R3 and R4 all survive. The other untested rules:
   - "Remove" only shows when a record is stored (connector-oauth-client-section.tsx:63, R8);
   - "Edit" only shows on a custom connector (connector-detail-page.tsx:74, R9);
   - the custom id pattern and the https endpoint (lib/connectors.ts:150,161, R11 and R12);
   - in the Connector filter the newest tick wins (connections-page.tsx:133, R13);
   - removing the End user filter resets the owner (connections-page.tsx:142, R18). If this one regresses, the page silently shows the signed-in user's connections instead of the app's;
   - View connections carries connector_id (connector-detail-page.tsx:70, R14);
   - an edit prefills the stored registration (lib/connectors.ts:193, R15).
   Fix: one test per rule that asserts on the request sent or on what is on screen.
2. [Should fix] lib/connectors.ts:198-217 + connector-dialog.tsx:53. Saving a revision drops the stored client.auth_method and client.alg. customConnectorValues and customConnectorBody carry only `registration`. A custom connector made through the API with an auth_method (connectors.go:296-297 accepts and stores it) loses it on "Save revision": the next revision falls back to the consent's own pick. Fix: carry connector.client.auth_method and alg through to the body when editing, and add a test.
3. [Should fix] connector-oauth-client-section.tsx:151-157 and connector-oauth-client-dialog.tsx:65-89 give false copy for provider-app-only connectors (linq, telnyx, whatsapp: registration [customer], schemes [bearer]):
   - The empty state says "connections use Stream's client … or register one at each consent". These connectors have neither operator nor dcr.
   - The dialog says "Leave empty only for a public client (none)". For these connectors both client_id and client_secret must stay empty.
   I saw this in the browser on /connectors/linq. Fix: build the empty-state copy from the registration (operator → Stream's client; dcr/cimd → per consent; otherwise "Connections need it"). When the connector lacks oauth2_code, say "provider app: ID and signing secret only".
4. [Nit] lib/connectors.ts:85,87 `.trim()` on client_secret and signing_secret. The router allows spaces in a secret (`^[ -~]+$`), so trimming changes the secret sent.
5. [Nit] The connections tab drops phase A's "End user = signed-in user" one-click owner switch. An End user filter now needs a typed ID (?owner=user still works).
6. [Nit] routes/.../library/connectors/index.tsx:22 prefetch, R19 (perf only).
7. [Nit] Test ids use a new AgentsConnectorsSelectors block. The other Library pages use AgentsLibrarySelectors. Kept as the coordinator directed.

## Author questions / spec gaps (honesty only)
- No DELETE for a custom connector: the UI has no delete button, so there is no dead button. OK.
- No endpoint in the response: the Edit dialog says "The router does not show the saved endpoint. Enter it again." Clear. OK.
- No redirect_uri: the setup steps for gmail, google_*, hubspot and shopify (router manifest text) say "the redirect URI shown on this page", and the page shows none. The UI cannot derive it, because AGENTS_ROUTER_URL is a proxy in dev. [Question → ticket]: expose redirect_uri on Connector as /plugins does (spec line 4427), or say so in the setup section.
- Provider app fields show for every connector. The spec does not expose channel.verifier, so the router's 400 is shown inline. Acceptable; folded into finding 3 only for linq-type copy.

## Design standards: compliant
- One Library > Connectors entry (sidebars.tsx:858-866, icon apps). No Resources group, route or test id is left over: a grep for agents/resources, AgentsResources and agents-resources is clean.
- Tabs use TabsProvider/TabHeader in Pane.Header `tabs`, as agent-page.tsx does. The heading "Connectors" follows the Voices pattern. Labels are in sentence case.
- dashboard/ tier only: no custom/ imports, no Remixicon or lucide, design-system Icon names, shared widgets (TableSection, DataTablePagination, TableErrorState, PrivilegeTooltipGuard, EntityDeleteDialog with confirmation). Filters use TableFilters in DataTable.Header and live in the URL. There is one component per new file. connection-detail-page.tsx keeps 6 components, from phase A audit finding 2, which is left to the splitting fixer.
- Secrets: the PasswordField defaults are '' and never prefilled. The section shows only Stored / Not stored. useAgentsMutation does not log the body. No console output except the existing PostHog warning.

## Mutations ($S/volt/rvb-mut.py, log $S/volt/rvb-mutations.log)
24 checked: a sample of 5 author mutations (all killed) and 19 of my own.
- Killed: R5, R6, R16.
- Survived: 16. They are R1-R4, R7, R8, R9, R10, R11, R12, R13, R14, R15, R17, R18 and R19.
- Not rules: R7 (ids are already [a-z0-9_]) and R17 (the schema's min(1) already clears it).
- Nits: R10 (the router accepts scopes for any scheme) and R19.
- Should fix: the rest (finding 1).
Re-run one mutation: `python3 $S/volt/rvb-mut.py <tree> R13`.

## Browser (:3014, tab 16, now closed)
- Catalog: 18 connectors, tabs, Add custom connector.
- linq detail: Overview, OAuth client empty state with the false copy (finding 3).
- Set OAuth client dialog opened and cancelled. Nothing was written.
- Connections tab, with and without End user.
- Connection detail at /library/connectors/connections/<id>. I created `ui-test-rvb` (github bearer, app, pending) with curl, saw it listed and its detail rendered (Status, Definition, Replace token, Validate), then deleted it: DELETE 204, GET 404.
- Console: only the PostHog warning.
