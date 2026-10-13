# Wave 9 current state (handoff detail), 2026-10-13

Do not commit this file. Wave 9 is done, deployed as `accelerate-v0.6.35` (`8160c4cd`). Exit tickets AI-1105..AI-1145 are under AI-816 (detail in Pending). The ledger is `9.md` (Decisions, rulings, PR table, Log, Tickets to file, Human gates). The orchestrator owns `9.md`. Do not edit it. This note is the short copy for the handoff. Plan: `~/.cursor/plans/connectors_hackathon_polish_2108a873.plan.md`.

Checked 2026-10-13 against `9.md` log 03:06Z (B3 GitHub PAT) and 03:05Z (B1 prep). Deploy facts below are still the 00:42Z log. No fresh `gh pr view` this sync. Live `/openapi` is 404, so the live spec is unverified. Volt phase J is wave 10 (`10-current.md`), not this note.

## agent.yaml `connectors:`

Go SDK on `29a014f0` (#895), `sdks/go/agents/folder.go` `ConnectorSettings` and `config.go` `connectorBinding`. Python is the same shape, merged in #896 (`c891381c`, head `daed2d08`). JS reads `connectors:` as of colleague #887 #898 #899 (on the v0.6.35 tag). Ruby, PHP, Rust, and .NET refuse it. Docs draft getstream.io#651 (GO at `1b65a4c`) names the readers: CLI, Go, Python, JS.

```yaml
connectors:
  - linear                       # session, tool_patterns: ["*"]
  - name: github
    connector: github            # defaults to name
    tools: [list_*, get_issue]   # sent as tool_patterns, as written
    connection: company          # id, or exactly one label -> fixed
    required: true               # omit to leave optional
plugins: [hubspot]               # still works; WARN, see below
mcp_servers:                     # unchanged
  - name: tablejourney
    url: https://tablejourney.com/mcp
```

- A string is a session binding. `tool_patterns` is `["*"]`. `tools` on the sync body is an empty list (`config.go` `connectorBinding` at `29a014f0`).
- An object is `{name, connector, tools, connection, required}`. `connector` empty means `name`. `tools` is copied into `tool_patterns`. `connection` set is fixed: an exact connection id wins, else exactly one label (P3). `required` nil stays optional.
- `tools` absent or null means every tool (`["*"]`). Comment on `ConnectorSettings.Tools`: "Nil, whether the key was left out or null, means every tool."
- `tools: []` is refused by the SDK: `connector %q: tools: [] grants nothing; leave tools out for every tool`.
- `plugins:` still syncs. The WARN text is `agent.yaml plugins are deprecated: rename plugins to connectors` (`config.go:69` at `29a014f0`, test `agent_test.go:612`).
- `mcp_servers:` is a separate key. Ledger decision: unchanged.
- A null list entry is refused (`a connector needs a name`). Unknown keys in an object are refused (`KnownFields`).

## Router `tool_patterns`

On `5857314d` (#893). Field `AgentConnectorBinding.ToolPatterns` (`configs.go` doc tag) and `store.ConnectorBinding.ToolPatterns` (`models.go`).

- Values are tool names or `path.Match` patterns (`plugins.CheckToolPatterns`, `plugins.Offered`).
- Exclusive with a non-empty `tools` (`configs.go` at `5857314d`: "a binding has one or the other"). A pattern binding sends `tools: []`.
- Allowed on session and on fixed. Pins are per `connection.ID` + `ConnectedAt`, so a fixed connection is trust-on-first-use too (ledger Decision; `patternGrants` calls `ConnectorToolPins` / `PinConnectorTools` with those two).
- Cap 128 (`maxItems`, `patternToolCap` in `session/connector_tools.go`). More than 128 matches: first 128 by name, one WARN.
- An empty `tool_patterns` list is refused (`minItems: 1`). Ruling P6 (log 16:55Z).
- Each match is pinned at the digest listed then. A later session offers it only while that digest still matches. A schema change waits for reconnect.
- Per call, `dispatcher.recheck` (`session/dispatcher.go` at `5857314d`) allows the tool while the current patterns still match it (`plugins.Offered`).
- Rollback to v0.6.34 fails closed. The field is JSONB. An older router drops the unknown key and reads `tools`, which a pattern binding leaves empty, so it grants none (`models.go` comment at `5857314d`; ruling P1). `tool_patterns` has no migration. Slot `20261018120000` is taken by #897 (rename of `20261015120000_custom_models`).

## When tools are listed

At each session open. `patternGrants` calls `Discover` (one `tools/list`), then `source.Open` lists again (`connector_tools.go` comment at `5857314d`: "The list is a separate tools/list from the one Open does"). Not mid-session. The MCP client sets `DisableStandaloneSSE: true` and does not read `notifications/tools/list_changed` (`sources/mcp/source.go` at `5857314d`).

## PRs (ledger, 2026-10-13)

| PR | State | SHA | What |
|---|---|---|---|
| GetStream/Vision-Agents#891 | MERGED | `06b7bce4` | AI-1078: validate answer drops the credential |
| #892 | MERGED | `dff5f98e` | shared-thread tests after ca8bcd5d |
| #893 | MERGED | `5857314d` | AI-1104: `tool_patterns` on bindings |
| #895 | MERGED | `29a014f0` | AI-1104: Go SDK `connectors:` |
| #896 | MERGED | `c891381c` | AI-1104: Python SDK + `examples/text_agents/mcp_plugins`. Head `daed2d08`. Ledger 18:12Z. `gh`: MERGED, mergeCommit `c891381c`. |
| #897 | MERGED | `8160c4cd` | w9-mig: move `20261015120000_custom_models` (`75214efe`) to `20261018120000`, idempotent DDL. GO was `2a04d8de`. Squash `8160c4cd` is the v0.6.35 tag. Ledger 00:03Z. |
| GetStream/cli#47 | MERGED | `eab135fa` | Merged into `beta` (squash `eab135fa`; GO was `c1e034a9`). Beta release `v2026-10-13-eab135f` is out (channel=beta; first dispatch failed on channel=stable). Log 01:37Z. |
| getstream.io#651 | MERGED | `f2211fa4` | Merged into `agents`. GO was `1b65a4c`. Readers: CLI, Go, Python, JS. Ruby/PHP/Rust/.NET refuse. Preview publish run 38332963443. Log 01:37Z. |

Router SHAs are from the ledger log 00:42Z and the deploy report. cli#47 and docs#651 are from the 01:37Z log. No fresh `gh pr view` this sync.

## Staging

`accelerate-v0.6.35` = `8160c4cd`, deployed 2026-10-13T00:12:48Z (router and gateway). Pre-tag goose gate passed (live was `20261016120000`). Goose is now `20261018120000`: 17 index swap (`20261017120000`), 18 custom_models (`20261018120000`). t0 and the 00:32Z recheck: 0 restarts, 0 ERROR, 0 5xx, `/health` 200. No rollback. Rollback target is v0.6.34 (starts on a DB at 18: no migrations to run). Ledger deployer `a690effe`.

`tool_patterns` is in the spec at `8160c4cd`. The live process does not serve `/openapi` (404), so the live spec is unverified.

Ships on the tag: wave 9 (#891 #892 #893 #895 #896), #897, colleague #887 #898 #899 (JS SDK reads `connectors:`), `75214efe` custom models.

Deploy rule still holds for the next tag: compare new migration files with the live goose version. No file below it.

Local: a DB that recorded `20261015120000` makes `router plugins migrate` refuse the renamed file. Delete that `goose_db_version` row. A branch that still carries `20261015120000` fails on a DB migrated by #897 until it merges `accelerate`.

The 3 ca8bcd5d session tests are fixed by #892 (ledger log 16:50Z: they pass). They are not known failures. `9.md` "Known base failures" still names them (baseline snapshot). Ignore that list for these three:

- `TestAttachConnectorsSuite/TestASharedConversationUsesTheAppsConnectionsOnly`
- `TestAttachConnectorsSuite/TestNoSelectionImpliesNothingForAnUnverifiedCallerOrASharedConversation`
- `TestConnectorLoginSuite/TestOnlyAnOptionalBindingOfTheCallersOwnWaitsForALogin`

Other known failures stay the `9.md` list (gofmt parse error is `internal/api/streams_test.go:417`, not `:382`).

## Pending for Kanat

CLI beta `v2026-10-13-eab135f` is released. Docs getstream.io#651 is merged (`f2211fa4`). Volt phase J is done: `10-current.md`.

Superseded 2026-10-13 18:20Z. The vendor logins and the staging secrets are done. Do not reopen the 03:05Z list. Continuation is `_connectors_handoff.md` section 7: deploy #905 (`cfd530953d`, not on staging), then Shopify scheme `none` on `stream-agents-e2e.myshopify.com`; B3 tool calls on the 11 Connected rows; Gong and Instacart stay blocked; do not post the Thierry reply.

Wave 9 is done. Exit tickets AI-1105..AI-1145 under AI-816. AI-1129 (#892) and AI-1142 (#897) are Done. AI-1120 is Done in wave 10. AI-1145 is the #897 M8 test gap. AI-1121 (publish docs#651) was the docs merge; ticket state not rechecked.

## Rulings (do not re-open)

P1-P5 are the block at the top of `9.md`. P6 and P7 were made during the wave (Log):

- P1. No migration for `tool_patterns`. Rollback to v0.6.34 fails closed. Slot `20261018120000` is #897, applied on staging (goose 18).
- P2. The `mcp_plugins` example ships in the Python PR, not the Go PR.
- P3. `connection: <x>` is resolved in the SDK at sync. Id wins, else one label. 0 or more than 1 is an error. No router change.
- P4. Session open discovers, matches, pins unpinned matches, then `Open`. Cap 128.
- P5. `tools` absent means `["*"]`. `tools: []` is refused.
- P6. The router refuses `tool_patterns: []` (`minItems: 1`). Log 16:55Z.
- P7. Company accounts in the example are fixed bindings via `connection: <label>`. The README names the label. Log 17:21Z. Landed in #896 (`c891381c`).
