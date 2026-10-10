# Phase B author log (2026-10-09)

PR: https://github.com/GetStream/volt-dashboard/pull/996 (draft), branch `connectors/agents-ui-b`, head 0096df3f42 (2 commits), base `connectors/agents-ui` 744c52f775. Worktree $S/volt/wt-b. PR body: $S/volt/pr-b/body.md.

## Course corrections applied
1. Coordinator: no one-entry Resources group -> moved under Library.
2. Coordinator (Kanat): ONE nav entry Library > Connectors at agents/library/connectors/, two tabs (Connectors, Connections) via TabsProvider/TabHeader in Pane.Header; connection detail at agents/library/connectors/connections/$connectionId; test ids `agents-connectors__*` (AgentsConnectorsSelectors); phase A's `AgentsResourcesSelectors` removed, `connectionsUserInput` dropped (input replaced by TableFilters).
3. Delete dialogs pass `confirmation` (OAuth client remove: type connector id). Filters in TableFilters inside DataTable.Header (catalog q; connections End user + Connector).
4. connection-detail-page.tsx touched only for the route strings (2) and the test-id import (the route moved); left for the splitting fixer otherwise.

## Files
- lib/connectors.ts: paths, search schema, takesOwnClient, isOwnClient, labels, oauthClientSchema/oauthClientBody, customConnectorSchema/Values/Body.
- dialogs/connector-dialog.tsx (create / next revision), connector-oauth-client-dialog.tsx (PUT), connector-oauth-client-remove-dialog.tsx (DELETE, EntityDeleteDialog confirmation).
- sections/connector-overview-section.tsx, connector-setup-section.tsx (Steps), connector-oauth-client-section.tsx.
- widgets/connectors-layout.tsx (tabs), connectors-catalog.tsx, connector-create-button.tsx, connector-detail-page.tsx; connections-page.tsx now content-only + TableFilters.
- routes library/connectors/{index,$connectorId}.tsx, connectors/connections/{index,$connectionId}.tsx (git mv from resources/connections). routeTree regenerated.
- tests/unit/agents/connectors-page.test.tsx (16, fetch stubbed under real agentRequest); connections-page.test.tsx paths + URL-based end-user test.

## Evidence
- tsc -p tsconfig.app.json: 1121 errors head == 1121 base (wt-int @744c52f775), diff of sorted error lines empty.
- bun lint 0; knip:check 0 (exports 316); build 0; vitest unit 232 files / 2631 tests pass (needs .env.local copied, else VITE_AMPERE_API errors).
- Mutations: $S/volt/pr-b/mutations.log (script $S/volt/pr-b/mut.py). All 12 rules fail >=1 test. M8 (explicit onDismiss after save) dropped: FormDialog closes itself on resolve, so not a rule.
- Local router via dev proxy :3011 (/__agents/1181507/1257545): connectors?limit=25 200 (18), &q=slack 200 (2), connectors/slack_bot 200, slack_bot/oauth-client 404, connections?owner_type=app 200 (0). Vite served all new modules 200.
- Direct curl X-Customer-Id 1257545: PUT github oauth-client {client_id: ui-test-client, secret} 201; GET has_client_secret true (no secret, no fingerprint); PUT without secret 400 "auth_method client_secret_post needs client_secret"; DELETE 204; GET 404 (restored). PUT linear 400 (registration dcr). POST oauth2_code w/o registration 400. Deployment schemes: api_key, bearer, none, oauth2_client_credentials, oauth2_code (from a 400 probe, nothing created).
- No custom connector created: the API has no DELETE for one.

## Gaps / questions
- No DELETE /v1/agents/connectors/{id} on origin/accelerate 26051062: custom connectors cannot be deleted (UI has none).
- Connector response has no endpoint: Edit asks for it again.
- API returns has_client_secret / has_signing_secret only, no fingerprint.
- Provider app fields shown for every connector that takes a customer client; the router refuses signing_secret for one without channel.verifier provider_app (400 shown in the dialog). The Connector response does not expose that, so the UI cannot hide them.
- Old resources/connections URLs get no legacy redirect: they only existed on the integration branch.

## Browser check (Fast UI loop)
- chrome-devtools tab on https://local.getstream.io:3011 (signed in after Kanat logged in; first attempt hit /login, shot b-login.png).
- Shots in $S/volt/shots/: b-catalog, b-connector-detail (github), b-oauth-client-dialog (before the select fix: shows "Select"), b-oauth-client-stored, b-oauth-client-remove, b-connector-dcr (linear), b-connector-setup (gmail steps), b-connections-tab, b-custom-connector-dialog.
- github: Set ui-test-client + secret -> Stored; Remove (typed github) -> 404 state again. Custom connector dialog not submitted (no delete API).
- Console: only PostHog apiKey warning (pre-existing).
- Fix commit 0096df3f42: auth-method select default 'default' option; overview row renamed "Client registration". Mutation M14 ('' default) fails 2 tests.
- Gap: Connector has no redirect_uri though gmail/google_*/hubspot setup steps say "redirect URI shown on this page".
- Dev server stopped; certs and .env.local removed.
