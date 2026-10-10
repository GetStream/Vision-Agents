# Phase G author log (2026-10-09)

PR: https://github.com/GetStream/volt-dashboard/pull/1000 (draft). Branch connectors/agents-ui-g, head db69ed017d, base connectors/agents-ui 2a5e3f0a19 (unchanged at push). PR body: $S/volt/pr-g/body.md.
Worktree: $S/volt/wt-g. Router facts read at Vision-Agents origin/accelerate 599298c6.

## Root cause of F60 (router evidence)
- validateConnection -> validationAfter (api/connection_tools.go:395-482) returns status=failed for any non-401 failure and leaves the row connected. Only Resolver.Invalidate on a 401 sets needs_reauthorization + last_error (connectors/resolver/resolver.go:199-220).
- GitHub MCP answers a wrong token with 400 "Bad Request" on initialize (e2e-d2 notes 23:30:14Z; re-seen live here, connection 209cf95b). So there is no 401 and no state change.
- The Connection API has no last_error / last-validation field (api/connections.go:38-58). The dashboard therefore keeps the validate answer in the query cache, keyed by connection and owner, with the credential revision it checked. QUESTION for the router: persist it.

## Files
- lib/connections.ts: CONNECTION_LISTS, ownerLabel, LastValidation, lastValidationQueryOptions (enabled:false, gcTime Infinity), currentValidation (revision guard), validationFailed (needs_scopes counts as works), lastCheckLine.
- lib/use-connection-validate.ts (new): POST validate, then fetchQuery of the connection to learn the revision (an OAuth renew during validate moves it), then setQueryData.
- lib/use-agents-mutation.ts: onSuccess may return a promise. It is returned, so mutateAsync awaits it. Type is unknown, because callers return values.
- dialogs/connection-create-dialog.tsx: after the PUT, made.current = stored (retry uses the new revision), then validate. failed/needs_reauthorization/pending -> throw inline. A validate request error -> proceed. The success toast moved after the check.
- dialogs/connection-credentials-dialog.tsx: onReplaced(stored) -> the page validates. Copy changed to "It is checked with the provider once it is saved".
- widgets/connection-overview.tsx + connections-page.tsx: "Last check: X" under the status. useQueries over the rows.
- widgets/connection-detail-page.tsx: uses the hook and the cache. On delete: navigate, then removeQueries for path, path/*, last-validation.
- dialogs/connection-delete-dialog.tsx: invalidate CONNECTION_LISTS.
- lib/connectors.ts keepsOAuthClient. The section's query is enabled only then. The route loader prefetches the connector first, then the oauth-client only if kept.
- lib/connector-bindings.ts bindingBreak / bindingBreakMessage. sections/agent-connectors.tsx: useQueries of the fixed connections (label F64, 404 -> broken), Broken tag, Choose connection (dialog gets a copy with the connection cleared), connector-gone has Remove only.
- dialogs/connector-binding-dialog.tsx F65 copy. dialogs/agent-delete-dialog.tsx F68: reads each fixed connection's used_by while open.

## Checks
- tsc -p tsconfig.app.json: base 1124, head 1124, diff empty (g-tsc-base.txt, g-tsc-head.txt).
- bun lint 0 (g-lint.log); knip:check 0 (g-knip.log); build 0 (g-build.log); unit 236 files / 2739 tests pass (g-unit.log, with .env.local copied in, now removed).
- New tests: connection-create.test.tsx (+12: provider refusal, retry revision, needs_scopes, validate 500, last check page/list, replace validates, revision guard, renew, delete re-read, owner x2), connectors-page.test.tsx (+3 incl. route loader), agent-connectors.test.tsx (+9 incl. F68 x2). One existing expectation changed: "App connection gh-app" -> "App connection Org bot" (F64). The first New connection test now also expects the validate POST.

## Mutations ($S/volt/g-mut.py, log $S/volt/g-mutations.log): 26/26 killed
- 0 F60 New connection checks the stored token: exit=1 failed=['a token the provider does not take > keeps New connection open with the provider’s reason, and Cancel deletes the connection', 'a token the provider does not take > stores another token on the revision the first one made, and opens the connection once it works']
- 1 F60 retry stores on the revision the first try made: exit=1 failed=['a token the provider does not take > stores another token on the revision the first one made, and opens the connection once it works']
- 2 F60 a validate the router cannot run does not block: exit=1 failed=['a token the provider does not take > opens the connection when the router cannot run the validate']
- 3 F60 needs_scopes is a working credential: exit=1 failed=['a token the provider does not take > opens the connection when the token works but lacks scopes a tool needs']
- 4 F60 Replace token is validated: exit=1 failed=['a connection’s last check > is checked again when the token is replaced, and says what it found']
- 5 F60 last check under status on the page: exit=1 failed=['New connection > creates an app token connection, then stores the token on it alone', 'a connection’s last check > is checked again when the token is replaced, and says what it found', 'a connection’s last check > is kept for a grant the validate itself renewed', 'a connection’s last check > is no longer shown once the credential it checked was replaced', 'a connection’s last check > says a failed validate under the status, on the page and in the list', 'a token the provider does not take > opens the connection when the token works but lacks scopes a tool needs', 'a token the provider does not take > stores another token on the revision the first one made, and opens the connection once it works']
- 6 F60 last check in the list: exit=1 failed=['a connection’s last check > says a failed validate under the status, on the page and in the list']
- 7 F60 answer dropped once the revision moves: exit=1 failed=['a connection’s last check > is no longer shown once the credential it checked was replaced']
- 8 F60 revision read after the validate: exit=1 failed=['a connection’s last check > is kept for a grant the validate itself renewed']
- 9 F61 delete invalidates lists only: exit=1 failed=['deleting a connection from its page > reads none of its own reads again once it is gone']
- 10 F61 deleted connection reads removed: exit=1 failed=['deleting a connection from its page > reads none of its own reads again once it is gone']
- 11 F62 section skips the read: exit=1 failed=['OAuth client reads > reads none for a connector the router keeps no client record for']
- 12 F62 loader skips the read: exit=1 failed=['OAuth client reads > is warmed by the page’s loader only for a connector that can have one']
- 13 F62 managed record still read: exit=1 failed=['OAuth client reads > reads the one Stream keeps for a connector that takes only Stream’s']
- 14 F63 deleted connection detected by 404: exit=1 failed=['a binding to what was deleted > is marked broken when its connection is gone, and can be pointed at another']
- 15 F63 partial catalog proves nothing: exit=1 failed=['a binding to what was deleted > is not judged gone from a catalog page that is not the whole catalog']
- 16 F63 deleted connector detected: exit=1 failed=['a binding to what was deleted > is marked broken when its connector is gone, and can only be removed', 'a binding to what was deleted > is not judged gone from a catalog page that is not the whole catalog']
- 17 F63 repoint drops the deleted connection: exit=1 failed=['a binding to what was deleted > is marked broken when its connection is gone, and can be pointed at another']
- 18 F63 no edit for a deleted connector: exit=1 failed=['a binding to what was deleted > is marked broken when its connector is gone, and can only be removed']
- 19 F63 broken tag: exit=1 failed=['a binding to what was deleted > is marked broken when its connection is gone, and can be pointed at another', 'a binding to what was deleted > is marked broken when its connector is gone, and can only be removed']
- 20 F64 fixed binding named by label: exit=1 failed=['a binding to what was deleted > is not broken while its connection and connector are there', 'connector bindings on the tools tab > saves a fixed binding with each granted tool pinned to its digest']
- 21 F65 lists-none copy: exit=1 failed=['what the tools hint says with no tools to check > says the chosen connection lists none']
- 22 F65 no-source copy: exit=1 failed=['what the tools hint says with no tools to check > says no connection was found to read them from']
- 23 F66 signed-in owner label: exit=1 failed=['a connection’s owner > is the signed-in user on their own connection, in New connection’s words']
- 24 F68 only when no other config binds it: exit=1 failed=['deleting an agent bound to an app connection > says nothing more when another config binds it too']
- 25 F68 warning shown: exit=1 failed=['deleting an agent bound to an app connection > says no other config answers on a connection only this one binds']

## Browser (:3011, own tab pageId 19, now closed; dev server PIDs 55275/55281 stopped; certs/.env.local removed)
1. Connections list: only ui-test-slackbot (Kanat's), unchanged.
2. New connection › GitHub › Bearer › App, label ui-test-g-bad-token, wrong token: POST 201, PUT 200, POST validate 200. The dialog stayed open with alert «The token did not work with the provider: mcp: connect to github: calling "initialize": sending "initialize": Bad Request. Enter another one, or cancel to delete the connection.» Screenshot taken.
3. A second wrong token: PUT 200 (revision correct), validate again, alert again. The list behind the dialog read «ui-test-g-bad-token … Connected Last check: Failed».
4. Navigated to its page (full reload, so the browser-held answer was gone, as expected). Validate -> alert «Validation: Failed / … Bad Request» and Status «Connected / Last check: Failed». Screenshot taken.
5. Agent ui-test-g-agent (6f3d9ec2…) via New agent (Custom, calls off). Tools › Add connector › GitHub › One app connection: hint before choosing «No connection to read its tools from yet…». After choosing ui-test-g-bad-token: «The connection has not listed its tools yet. Validate it on its page…» (F65). Row «App connection ui-test-g-bad-token · 0 tools» (F64). Saved.
6. Agent delete dialog (cancelled): «No other agent config binds ui-test-g-bad-token, so messages that arrive on it, such as a Slack bot’s, go unanswered.» (F68)
7. Connection page › Delete (typed): DELETE ?force=true 204. The next requests were only connector-audit, connectors, connections?owner_type=app: no 404 (F61).
8. Tools tab: «GitHub · github Broken / Its connection 209cf95b… no longer exists, so sessions open without it. Choose another connection or remove it to save the agent.» Screenshot taken. Choose connection -> editor, Connection «No app connections to this connector» (cleared). Cancelled. Remove + Save -> «Agent saved».
9. The agent was deleted (typed confirm). Its dialog then had no extra sentence (no bindings).
10. linear page: no GET …/oauth-client (F62); text «This connector does not take your own OAuth client. It uses: Registered at each consent.» slack_bot page still shows its stored client.
11. Kanat's Slack c574c80d page (read only): Owner «Signed-in user volt-1115938» (F66).
- Console: PostHog warn only.
- State at the end (GET via proxy): configs ["ui-test-slackbot-agent"] (made by Kanat during this run, not touched), app connections ["ui-test-slackbot"]. No ui-test-g-* left.

## Questions
- Router: persist the last validate (for example Connection.last_validation {status, code, error, checked_at}), or set status/last_error on a failed validate. Without it, "Last check" lives in one browser tab's cache until reload.
- Router: expose the manifest's channel block (for example Connector.channel: bool), so F68 can warn only for channel connectors. Today it warns for any fixed connection only this config binds.
- Known edge, not changed: leaving New connection by navigating (not Cancel) after a refusal leaves the connection. This was already true for the router-refusal path.
