# Reviewer C round 1 — volt PR #997 (head 9f137f5ca1, base origin/connectors/agents-ui 744c52f775)
VERDICT: NO-GO (2 Should fix)

## Setup
- Worktree $S/volt/wt-rc (detached 9f137f5ca1; `git merge origin/connectors/agents-ui` -> Already up to date). Base worktree $S/volt/wt-rc-base at 744c52f775 for tsc. Both removed at end.
- Router source: Vision-Agents 26051062 (ancestor of origin/accelerate; `git merge-base --is-ancestor`).

## Checks (logs in $S/volt/pr-c/rv-*)
- bun lint 0 (rv-lint.log); knip:check 0 (rv-knip.log); build 0 (rv-build.log).
- unit: 234 files / 2633 tests pass (rv-unit.log). NOTE: without .env.local copied, 27 files fail (auth/fetcher/moderation, env-dependent, unrelated); with it all pass.
- tsc -p tsconfig.app.json: base 1121 errors, head 1121; only diff is the pre-existing use-conversation.ts findLast error's printed union (rv-tsc-base.norm vs rv-tsc-head.norm).
- `bun run gen:agents`: exit 0, `git status --short` empty. api/agents/openapi.yaml byte-identical to `git show 26051062:acceleration/api/openapi.yaml` (diff -> SPEC-IDENTICAL).

## Binding rules vs router (26051062)
| Rule | Router | UI | Verdict |
|---|---|---|---|
| alias pattern + no `__` | configs.go:475, spec pattern | connector-bindings.ts:16,94 | match |
| unique alias | configs.go:479 | :99 | match |
| plugin alias unless own connector | configs.go pluginAliasComplaint (`binding.ConnectorID != binding.Name && namesPluginEntry`) | :103-104 | match |
| MCP server alias | pluginAliasComplaint | :104 | match, untested (R1 survives) |
| fixed needs connection_id; session refuses one | configs.go:487-495 | :106; writes `{type:'session'}` | match, :106 untested (R2) |
| fixed connection is app-owned + same connector | configs.go unboundConnectors | dialog lists owner=app & connector_id | match |
| events only on fixed | configs.go:497 | :108 + dialog passes !!binding.events | match |
| fixed grant needs digest | configs.go:520 | listed digest + unlisted throw (dialog:93-106) | match |
| session binding grants by name | pluginmigrate/AGENTS.md "A session binding grants by name… one person's digests would leave the others' tools unavailable" | bindingFromFields keeps `existing` digests for session (:154-158) | MISMATCH -> SF1 |
| timeout 1..30000 | spec min/max | :113-116 | match, max untested (R3) |
| tools maxItems 128 | spec | no cap | Nit (router 400 on save) |
| duplicate grant | configs.go:518 | ChipInput dedups (browser: typed list_issues twice -> 1 chip); checkbox group can't dup | ok |
| binding wins over plugin entry | session/spec.go withoutBoundPlugins/boundProvider | replacedPlugin/bindingReplacing tags | match |

## SF1 probe (scratch test in wt-rc, removed)
Fixed binding {post, digest a*64} -> edit, switch to "Each user's own" -> bindingFromFields output:
`{"connection":{"type":"session"},"tools":[{"name":"post","schema_digest":"aaaa…"}]}` (expected `[{name:'post'}]`). Router accepts it (digest optional for session) and then pins every user's connection to the app connection's digest.
Fix: in bindingFromFields, for a session binding never carry a digest (or only when existing.connection.type === 'session'); add a test switching fixed->session.

## Author Q1: test copy with event bindings — yes, and worse
- Copy = POST of agentRequestFromDraft (use-test-draft.ts:81-85; also agent-duplicate-dialog.tsx:62, agent-save-as-new-dialog.tsx:68 via newAgentRequest). connectors ride along (base via `...kept`; head via `connectors: draft.connectors`), so pre-existing in base.
- mcpevents.Reconcile (internal/mcpevents/mcpevents.go:210-227) adds a subscription per live config binding the connection fixed; store conflict key is (connection_id, config_id, binding, key) (internal/store/connection_events.go:76) -> the copy gets its own subscription at the next validate (api/connection_tools.go:402) or reconnect (api/authorizations.go:734): each event opens a conversation twice.
- Worse: channelbridge take() drops every inbound message when `len(configs) != 1` (internal/channelbridge/bridge.go:425-433; gate.go:64). A copy/duplicate with a fixed binding to a channel app connection (slack_bot etc.) silences the live agent while the copy lives (test copy up to STALE_COPY_MS 2 h; duplicate permanently).
- Ruling: ticket(s), pre-existing, needs a router decision (router does not know volt's draft_of tag). Router: refuse a 2nd config fixed-binding a channel connection, or channelbridge/mcpevents ignore draft copies. Volt: test copy strips events / fixed channel bindings.

## Author Q2: session tool names from app's or signed-in user's connection, else typed
OK as is (names only; session grants by name). No change.

## Consent card
- Only `^https://` launch pages (use-conversation.ts messageConnectorLogins); phase A startConsent: token only via postMessage to launch origin (connector-consent.ts:102-113).
- Browser: popup opened `https://carol-elliptic-uncloak.ngrok-free.dev/v1/agents/connectors/oauth/launch/<authorization id>` — no token in URL; local/sessionStorage scan for handoff|consent|oauth: none. Popup stopped on ngrok interstitial ERR_NGROK_6024 (phase A Question, phase D).
- Router attachment shape (internal/conversation/connector_logins.go:28-42) matches the zod branch; expires_at always sent.

## Browser (:3013, signed in)
- Created ui-test-rc (text), Tools tab: Connectors section + empty state render; Add connector dialog (labels above fields, radio descriptions); chose Linear, name auto "linear", typed tool; row "Linear · linear / Each user's own account · 1 tool"; Save -> GET config connectors `[{"name":"linear","connector_id":"linear","connection":{"type":"session"},"tools":[{"name":"list_issues"}],"required":false}]`.
- Linear app row (agent has NO linear plugin entry) shows "Sessions use the linear connector" next to "Connect" -> Nit N2.
- Playground: "List my Linear issues…" -> reply + Connect Linear BUTTON (data-testid agents-test__connector-connect-button, class ds-button--md, while plugin links are sm) -> Nit N1.
- Console: only PostHog warning. Cleanup: config deleted (204), list empty. Session 01a12280… left to time out (close endpoint 404).

## Mutations ($S/volt/pr-c/rv-mut.py, rv-mut.log): 19 run, 6 rule survivors (+R10 = SF1 inverse)
Author sample M1 M4 M6 M8 M9 M14 M16 M19: all KILLED.
Mine: R1 MCP alias SURVIVED; R2 fixed needs connection SURVIVED; R3 timeout max SURVIVED; R4 edit excludes itself KILLED; R5 agent-tabs connectors->tools SURVIVED; R6 connected tag after end KILLED; R7 no button without handoff_token SURVIVED; R8 app-owned fixed options KILLED; R9 aliasFor unique suffix SURVIVED; R10 session keeps no digest SURVIVED (no test pins either way = SF1); R11 create greeting object KILLED.
Re-run: `cd $S/volt/wt-rc && python3 $S/volt/pr-c/rv-mut.py R1 R2 R3 R5 R7 R9 R10`

## Design standards
dashboard/ tier only, no custom/ imports, no Remixicon, Icon from design-system (diff grep: none). Shared PrivilegeTooltipGuard reused. FormDialog/Form.* /Radio/Checkbox/SettingRow/EmptyState/Tag primitives. Test ids in test-ids.ts BEM incl. dynamic `--name` modifier (matches channel-types pattern). Sentence case OK. Nits: N1 button size, N3 Edit/Remove disabled without the tooltip guard Add has ("A disabled action needs a nearby reason", design-review).
