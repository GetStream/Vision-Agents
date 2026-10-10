# Phase H author log (2026-10-09/10)

PR: https://github.com/GetStream/volt-dashboard/pull/1001 (draft). Branch connectors/agents-ui-h, head 14248193dc, base connectors/agents-ui 0043c23d11 (unchanged at push). PR body: $S/volt/pr-h/body.md. Worktree $S/volt/wt-h.
Router facts at Vision-Agents 599298c6 (spec copy $S/volt/h-openapi.yaml).

## Scope as finally decided (coordinator messages during the run)
- Items 1, 2, 3 of phase-h.md. Plugins section («Connected apps») NOT merged and untouched (Kanat: «плагины не трогаем»). No plugin tag, no «Available apps».
- Connectors title «Connectors», description «Connectors this agent can use. Create new ones in Library › Connectors.» (link), search over bindings (connector name + alias), picker search in the add dialog.
- Library › Connectors: plain PaneSearchInput search (replaces TableFilters), Connector column links on Connections tab, scopes as Tags on connector and connection pages.

## Files
- dialogs/connector-binding-dialog.tsx: title, mode prop, picker search, Connect prompt (Alert) → ConnectionCreateDialog in FormDialogPortal, Advanced (TitleGroupExpandable) with typed names (fallback) + CallFields (voice only), consent → validate.
- dialogs/connection-create-dialog.tsx: `owner` prop ('me' default, 'app' for a fixed binding); copy «can use».
- lib/use-connector-consent.ts: connect(connection, popup?, onConnected?) — runs on status connected.
- lib/agent-draft.ts: text mode strips binding policy on save.
- sections/agent-connectors.tsx: copy, link, search, empty states, mode → dialog. widgets/agent-section.tsx: description ReactNode.
- widgets/connectors-catalog.tsx: PaneSearchInput + useDebounce → URL q; FilterableEmptyState.
- widgets/connections-page.tsx: Connector column RouterLink. sections/connector-overview-section.tsx, widgets/connection-overview.tsx: TagList of scopes.
- connector-create-button.tsx / connector-dialog.tsx: «New connector», «Create connector». connectors-catalog empty copy.
- No-bind copy: agent-delete-dialog, connection-delete-dialog, connector-delete-dialog, connection-overview («Used by», «No agent uses it…»).
- test-ids: AgentsToolsSelectors.connectorConnectButton.

## Checks
- tsc -p tsconfig.app.json: base 1120 / head 1120, diff empty ($S/volt/h-tsc-base.txt, h-tsc-head.txt).
- bun lint 0 ($S/volt/h-lint.log), knip:check 0 (h-knip.log), build 0 (h-build.log), unit 236 files / 2758 tests (h-unit.log).
- Existing tests changed: copy renames (New connector, Create connector, uses/Used by, dialog title); the policy test now uses a voice agent and opens Advanced; typed-name tests open Advanced; hint copies; connections-page link queries disambiguated (connectionLink helper); closed-dialog test catches its expected lookup rejection (it surfaced as an unhandled error once more tests ran after it).

## Mutations ($S/volt/h-mut.py, log $S/volt/h-mutations.log): 28/28 killed
(log note: first-run lines 22/23 were invalid mutations (syntax, SUITE FAILED); they were replaced by the four scope mutations in the rerun, numbered 22-25 there. First-run 24/25 are the library button and bind-copy mutations.)
- item2 text sends no policy → call settings > are never sent for a text agent
- item2 voice keeps policy → call settings > stay as they are for a voice agent (+ policy test)
- item2 call fields hidden for text → are not offered for a text agent
- item2 call fields shown for voice → are behind Advanced for a voice agent
- item1 typed names only under Advanced → names as many tools…, takes tool names under Advanced…
- item1 connect prompt → offers to connect the connector…, both connect tests
- item1 no prompt when a connection exists → has not listed / lists none
- item1 session owner signed-in user → makes a token connection for the signed-in user…
- item1 fixed owner App / connection_id set / consent then validate → makes an app connection by consent…
- item3 dialog title → 18 tests; section link, empty-state link → says what it is for…
- bindings search filter / alias match / filtered empty → narrows the agent’s connectors…
- picker search filter / empty → narrows the connectors the add dialog offers
- catalog search q / filtered empty → narrows the catalog…
- connector column link → links each connection’s connector…
- scope tags (one per scope, None) x2 pages → shows each (granted) scope as a tag…
- New connector button → 3 connectors-page tests; bind copy → says no other config answers…

## Browser (:3011, own tab pageId 22, now closed; dev server PIDs 73289/73323 stopped; :3000 PID 48732/48734 untouched; certs/.env.local removed; PAT helper killed, url file removed)
1. Library › Connectors: «New connector»; search «slack» → Slack, Slack bot, URL ?q=slack; «zzzz» → «No connectors match your search» (screenshot); Clear search → 18 rows, URL clean.
2. Connections tab: row ui-test-slackbot → name link to connections/c66c1de6…, «Slack bot» link to connectors/slack_bot/; click opened the connector page.
3. slack_bot Scopes: tags chat:write, channels:history, im:history (screenshot). linq Scopes: None. ui-test-slackbot connection Scopes: 3 tags.
4. Text agent ui-test-h-text (634686ee…): Tools › Connectors copy + both links (screenshot). Add connector → title «Add a connector to this agent»; search «git» → only GitHub option; GitHub → «Connect GitHub to choose its tools. [Connect]» + collapsed Advanced (screenshot). Connect → New connection (GitHub preset, Signed-in user checked) → Bearer, label ui-test-h-github, PAT via helper (never printed) → POST connections 201 (reqid 17335), PUT credentials 200 (17338), POST validate 200 (17341), GET tools 200 (17342/17345) → 49 tool checkboxes (screenshot), no Advanced (text + listing). Ticked add_issue_comment, get_me → row «Each user’s own account · 2 tools» → Save → «Agent saved» (PUT 17348). Stored: mode text, connectors [{github, session, tools [add_issue_comment, get_me], required false}] — no policy. Edit: no Advanced, no call copy, 2 checked.
5. Section search «zzz» → «No connectors match "zzz"»; «git» → the row (screenshot).
6. Voice agent ui-test-h-voice (23e16d70…): Add connector → Advanced present; opened → «What the agent says while a tool runs», «When the user interrupts»… (screenshot).
7. Console: PostHog warn; a 400 on GET /v1/agents/voices/library from the Behavior tab (not this PR's code); 403s preserved from earlier navigations (none in the resource list of the final page).
8. Cleanup via proxy: DELETE both configs 204, DELETE connection d168e61a…?force=true 204. After: ui-test configs ["ui-test-slackbot-agent"], app connections ["ui-test-slackbot"], signed-in user's [c574c80d slack, 91d5e6f9 linear] (Kanat's, untouched).
- Not done live: the OAuth consent path from the dialog (popup consent only completes from :3000); covered by the unit test with a real BroadcastChannel outcome.

## Questions
- Text agents now lose a saved binding `policy` (incl. pre_speech, which the router still reports on tool_started for any session) on their next save from the dashboard. Per Kanat's decision; flagging that it rewrites YAML-made text configs.
- The catalog search box replaces TableFilters, so the table's refresh button is gone (sibling Library pages have none either).
