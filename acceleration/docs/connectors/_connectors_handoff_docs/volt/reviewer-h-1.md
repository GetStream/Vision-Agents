# PR #1001 phase H — round-1 review (2026-10-09)

VERDICT: NO-GO (2 Should fix)

Reviewed head 1655836c9b, base connectors/agents-ui 0043c23d11 (merge of current base into scratch tree: "Already up to date").
Worktree $S/volt/wt-rh (removed at the end). PR is MERGEABLE on GitHub; CI lint/knip/unit/Vercel SUCCESS; 0 reviews, 3 bot comments.

## Checks (merged tree)
- bun lint 0 ($S/volt/rh-lint.log), knip:check 0 (rh-knip.log), build 0 (rh-build.log).
- unit: 236 files / 2760 tests passed (rh-unit.log).
- tsc -p tsconfig.app.json --noEmit: base 1124 / head 1124 errors; diff of sorted error lines (positions stripped) empty (rh-tsc-base.txt, rh-tsc-head.txt, rh-tb.s, rh-th.s). Author's 1120 figure differs by env; delta is what counts.
- Plugins («Connected apps»): no plugin file in `git diff --stat origin/connectors/agents-ui...HEAD`; agent-tools.tsx untouched. AgentSection.description widened to ReactNode (type only).
- «bind» in UI copy: python scan of every string literal / JSX text in src/components/dashboard/agents and agents routes (comments stripped) — 0 hits (only code identifiers).

## Findings
1. [Should fix] connectors-catalog.tsx:37-52 — the search box keeps its own `query`, read from the URL once. When q changes from outside, the effect writes the stale text back. Live on :3014: typed «slack» → ?q=slack; clicked sidebar «Connectors» (href has no q) → URL stayed ?q=slack, input «slack», 2 rows. Base (TableFilters reading search.q) cleared. Fix: follow search.q when it changes not from this box (e.g. reset `query` when search.q differs from `wanted`), with a test for the nav-link case and for a deep link ?q= kept (R18 survives).
2. [Should fix] tests missing for new rules (surviving mutations, $S/volt/rh-mutations.log, rh-mutations2.log):
   - R2 use-connector-consent.ts:86 `if (status === 'connected')` dropped → denied/failed consent would validate a pending connection. Test: a denied outcome sends no /validate.
   - R4 connector-binding-dialog.tsx:199 the chosen connector stays offered while the search excludes it.
   - R5 :229 only an app-owned connection becomes a fixed binding's connection_id (owner switched to Signed-in user in the New connection dialog).
   - R7 :225 fixed binding with app connections, none chosen → «Choose a connection to see its tools.» and no Connect prompt.
   - R8 :382 Advanced opens by itself when an edited binding has typed grants.
   - R9 agent-connectors.tsx:124 no search box without bindings.
   - R14 connectors-catalog.tsx:38 search text trimmed before it reaches q.
   - R18 connectors-catalog.tsx:37 deep link ?q=git keeps q (live it does; no test).
   R19 (drop the `wanted === search.q` guard) also survives; equivalent mutation (replace-navigate to the same search), not counted.
3. [Nit → ticket] Second Save on the agent page after a first Save in the same edit raised «This agent changed … saved at 9:10 PM after you started editing» for my own save (agent page save path, not touched by this PR; not checked on base) — unverified whether base does the same.

## Author QUESTIONS
- Text agents losing a saved policy: resolved by 1655836c9b. Verified live: text agent with policy {pre_speech:' One moment. ', on_interrupt:'wait', cancellable:true}; Edit → no call settings, toggled Required, Done, Save → stored policy byte-for-byte equal, required:true.
- Catalog refresh button gone with TableFilters: accept (sibling Library pages have none), no fix.

## Focus items
1. Connect from binding dialog: reuses ConnectionCreateDialog (phase E) + useConnectorConsent; token path validates in the create dialog, OAuth path validates on 'connected'. Saved bodies: session = names only, no digest (live + test); fixed = digest pinned (test, made-1 / d-find). Typed names only under Advanced (typed = connectorId && !listing).
2. Call settings only for voice (voice prop + CallFields); text keeps saved policy (bindingFromFields voice=false → savedPolicy); new text binding sends no policy (test + live GitHub binding saved without policy). timeout_ms stays visible for both (not one of the three call settings in the spec).
3. Copy: «New connector»/«Create connector», «Add a connector to this agent», «Connectors this agent can use. Create new ones in Library › Connectors.» with link, empty state link «Create one in Library › Connectors», «Search connectors» x3, «Used by». Sentence case OK.
4. Design: DS Alert/Button/TitleGroupExpandable/Tag/TagList, shared PaneSearchInput + FilterableEmptyState (same as agent-tools «Search apps»), RouterLink; no Remixicon; test id in test-ids.ts (BEM). The design system has no searchable Select (no Combobox/searchable in vendor/design-system), so a search box above the select is justified. Connections rows have no row click (no onRowClick in connections-page.tsx), so the Connector link cannot trigger a row.

## Mutations
Author sample re-run (rh-author-mut.py = h-mut.py pointed at wt-rh) indices 2 4 5 9 10 14 17 19 21 22 24 27: 12/12 killed. (Author 0/1 targeted agent-draft.ts, removed by the fix; replaced by my R0/R1.)
Mine (rh-mut.py, R0-R19): 20 run, 11 killed, 8 survived (R2 R4 R5 R7 R8 R9 R14 R18), 1 equivalent (R19).
Re-run: `cd <worktree> && python3 $S/volt/rh-mut.py [idx...]` (W in the script points at $S/volt/wt-rh/; recreate the worktree first). Author's: `python3 $S/volt/rh-author-mut.py [idx...]`.

## Browser (:3014, own tab 23, closed)
- Catalog: «New connector»; search slack → Slack, Slack bot, ?q=slack; sidebar Connectors link → q not cleared (finding 1). Deep link ?q=git → input git, GitHub only.
- Connections tab: ui-test-slackbot row; Connector «Slack bot» → /connectors/slack_bot/; click opened the connector page. Scopes: tags chat:write, channels:history, im:history (screenshot).
- Text agent ui-test-rh-text (41c1c3a4…): section copy + link; Edit linear → no call settings, Advanced open with typed list_issues; Save → policy byte-for-byte. Add connector → search «git» → only GitHub option; GitHub → «Connect GitHub to choose its tools. [Connect]» (screenshot); Connect → New connection, Signed-in user preset, Bearer, label ui-test-rh-github, PAT via loopback helper (never printed) → POST connections 201 (reqid 11577), PUT credentials 200 (11580), POST validate 200 (11583), GET tools 200 (11584) → 49 tool checkboxes, no Advanced, no call copy; ticked get_me, add_issue_comment → Save → stored {github, session, tools [get_me, add_issue_comment], required false}, no policy.
- Voice agent ui-test-rh-voice (06696368…): Add connector → GitHub lists tools; Advanced (collapsed) → the three call settings (screenshot).
- Console: PostHog warn, 403s preserved from earlier navigations; no 4xx resource on the final page.
- Cleanup: DELETE connection 731bd571…?force=true 204, DELETE both configs 204. After: ui-test configs [ui-test-slackbot-agent], app connections [ui-test-slackbot], signed-in user's [c574c80d slack, 91d5e6f9 linear]. Dev server PID 81771 (+child 81797) and helper 82677 stopped; :3000 PID 48734 untouched.

## UNVERIFIED
- OAuth consent from the binding dialog live (popup consent completes only from :3000); covered by unit test with a real BroadcastChannel.
- Finding 3 on base.
- /pr-review skill not run: it is the Vision-Agents review skill (OpenAPI/migrations); review done by hand against volt rules. No migration in this PR.
