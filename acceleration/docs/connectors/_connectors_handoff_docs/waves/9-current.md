# Wave 9 current state (handoff detail), 2026-10-12

Do not commit this file. Wave 9's five PRs are merged. Staging deploy stopped before tagging. The ledger is `9.md` (Decisions, rulings, PR table, Log, Tickets to file, Human gates). The orchestrator owns `9.md`. Do not edit it. This note is the short copy for the handoff. Plan: `~/.cursor/plans/connectors_hackathon_polish_2108a873.plan.md`.

Checked 2026-10-12 against `9.md` (log after 18:04Z, PR table, Human gates, Tickets) and `gh pr view 896 897 --json state,mergeCommit,headRefOid`. Live staging was not queried.

## agent.yaml `connectors:`

Go SDK on `29a014f0` (#895), `sdks/go/agents/folder.go` `ConnectorSettings` and `config.go` `connectorBinding`. Python is the same shape, merged in #896 (`c891381c`, head `daed2d08`).

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

## PRs (gh, 2026-10-12)

| PR | State | SHA | What |
|---|---|---|---|
| GetStream/Vision-Agents#891 | MERGED | `06b7bce4` | AI-1078: validate answer drops the credential |
| #892 | MERGED | `dff5f98e` | shared-thread tests after ca8bcd5d |
| #893 | MERGED | `5857314d` | AI-1104: `tool_patterns` on bindings |
| #895 | MERGED | `29a014f0` | AI-1104: Go SDK `connectors:` |
| #896 | MERGED | `c891381c` | AI-1104: Python SDK + `examples/text_agents/mcp_plugins`. Head `daed2d08`. Ledger 18:12Z. `gh`: MERGED, mergeCommit `c891381c`. |
| #897 | OPEN | head `2a04d8de` | w9-mig: move `20261015120000_custom_models` (`75214efe`) to `20261018120000`, idempotent DDL. Ledger 19:13Z GO. `gh`: OPEN, mergeCommit null, head `2a04d8de`. Merge waits for Kanat. |
| GetStream/cli#47 | OPEN | head `c1e034a9` | `agent.yaml`: accept `connectors` and `mcp_servers`. Ledger 17:44Z GO at `c1e034a9`. Held until staging has `tool_patterns`. Beta release needs Kanat. |

SHA prefixes match `gh pr view 896 897 --json state,mergeCommit,headRefOid` (cli#47 GO is the ledger, not a review re-read this sync).

## Staging

Still `accelerate-v0.6.34` (`e464eeeb`, goose `20261016120000`). Ledger 18:18Z: staging-deployer stopped before tagging. Staging unchanged. Live cluster not rechecked.

`accelerate` at `c891381c` includes Thierry's `20261015120000_custom_models` (`75214efe`, merged after #874), below the live goose version. goose has no out-of-order, so a tag would crash-loop. #897 moves that file to `20261018120000` with idempotent DDL (fresh, staging-shape, and already-applied-old). GO at `2a04d8de`. `accelerate-v0.6.35` is still free. Merge, tag, and deploy wait for Kanat. After the new router starts, rollback to v0.6.34 is fine (ledger 18:50Z: rollback starts, no migrations to run).

Deploy rule: before tagging, compare new migration files with the live goose version. No file below it.

Local: a DB that recorded `20261015120000` makes `router plugins migrate` refuse the renamed file. Delete that `goose_db_version` row. A branch that still carries `20261015120000` fails on a DB migrated by #897 until it merges `accelerate`.

The 3 ca8bcd5d session tests are fixed by #892 (ledger log 16:50Z: they pass). They are not known failures. `9.md` "Known base failures" still names them (baseline snapshot). Ignore that list for these three:

- `TestAttachConnectorsSuite/TestASharedConversationUsesTheAppsConnectionsOnly`
- `TestAttachConnectorsSuite/TestNoSelectionImpliesNothingForAnUnverifiedCallerOrASharedConversation`
- `TestConnectorLoginSuite/TestOnlyAnOptionalBindingOfTheCallersOwnWaitsForALogin`

Other known failures stay the `9.md` list (gofmt parse error is `internal/api/streams_test.go:417`, not `:382`).

## Pending for Kanat

- Merge #897, then tag `accelerate-v0.6.35` and deploy. Do not tag `c891381c`.
- CLI beta release (cli#47 GO at `c1e034a9`, held until staging has `tool_patterns`).
- Docs publish (plan track C: quickstart and creating-an-agent).
- Vendor consents / operator OAuth clients (plan B1). Prep notes: `migrate/operator-clients.md`. Slack on staging is still F87.
- Gong and Instacart: unverified. No account check this sync. Plan B3 flags Gong (customer or DCR; Stream may have no account) and Instacart (API key).

## Rulings (do not re-open)

P1-P5 are the block at the top of `9.md`. P6 and P7 were made during the wave (Log):

- P1. No migration for `tool_patterns`. Rollback to v0.6.34 fails closed. Slot `20261018120000` is now #897.
- P2. The `mcp_plugins` example ships in the Python PR, not the Go PR.
- P3. `connection: <x>` is resolved in the SDK at sync. Id wins, else one label. 0 or more than 1 is an error. No router change.
- P4. Session open discovers, matches, pins unpinned matches, then `Open`. Cap 128.
- P5. `tools` absent means `["*"]`. `tools: []` is refused.
- P6. The router refuses `tool_patterns: []` (`minItems: 1`). Log 16:55Z.
- P7. Company accounts in the example are fixed bindings via `connection: <label>`. The README names the label. Log 17:21Z. Landed in #896 (`c891381c`).
