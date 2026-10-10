# Connectors (AI-816) handoff, 2026-10-10

> **Do not commit this file or `_connectors_handoff_docs/`.** The repo is public. They are working notes. Keep GCP project, cluster and secret names out of anything you commit; those details live only in the memory file `staging-migration-order.md`.

You are taking over from the orchestrator agent that ran the AI-816 connectors work for Kanat (kanat.kiialbaev@getstream.io). Read this whole file before you act. Reply to Kanat in **Russian**, keep English code names untranslated, use they/them for people, and be brief.

## 1. What the project is

**Connectors** replace the old **plugins** (Connected apps). A **connector** is a service an agent can use: a built-in (Slack, Slack bot, GitHub, Linear and 14 more, defined as manifest YAML in the router) or a custom MCP server one app adds. A **connection** is an account connected to a connector, owned by the app or by one end user. A **binding** gives an agent config a connector's tools: `fixed` (one app connection) or `session` (each user's own account, consent in the Playground card). **Channel connectors** (slack_bot, linq, telnyx, whatsapp) also receive inbound messages.

Inbound flow today: provider event → router `/v1/connectors/events/{connector}/{provider_app}` → the router writes the message into a thread channel in the app's **Stream Chat** → Stream fires `message.new` to the router's hook `/v1/chat/hooks/stream[/<app id>]` → the agent replies. The planned rework (router hands the message to the agent directly, router-side queue, `message.new` becomes the ACK) is **AI-1077**, assigned to Kanat.

Repos and branches:

| Repo | Path | Branch | What |
|---|---|---|---|
| GetStream/Vision-Agents (public) | `~/Projects/stream/Vision-Agents` | `accelerate` (base for all router PRs) | Go router in `acceleration/`, SDKs, skills |
| volt-dashboard (private) | `~/Projects/stream/volt-dashboard` | `ai-team/agent-dashboard` | The product dashboard with the Agents section and the connectors UI |
| GetStream/chat | `~/Projects/stream/chat` | `thierry/accelerate-gke` | Staging gateway proxy (`projects/stream-accelerate/proxy.go`), helm chart, staging values, `rocky` CLI |

Branch naming: router PRs use `connectors/<slug>` off `accelerate`. The volt integration branch `connectors/agents-ui` is merged (#1002) and gone from origin; start new volt phases from `origin/ai-team/agent-dashboard`.

## 2. How the work is run (process)

Kanat wants the agent to **orchestrate, not code**. Authors write PRs, independent reviewers give GO or NO-GO, fixers fix, delta reviewers re-check. The orchestrator merges on GO only, then tags, deploys to staging and verifies.

- Skills: `plan-waves` (turn tickets into a wave ledger) and `run-wave` (execute it) in `~/.claude/skills/`. Read `run-wave` before running anything. It has the loop, the severity rules, the merge rules, and the lesson that GitHub ignores `merge=union`.
- Static subagents in `~/.claude/agents/`: `wave-author`, `wave-reviewer`, `wave-delta-reviewer`, `wave-fixer`, `wave-mechanic`, `e2e-tester` (local only), `staging-deployer`, `ticket-filer`.
- A wave's only state is its ledger `waves/<N>.md` (see §6). Update it on every state change.
- Merge with `gh pr merge N --squash --match-head-commit <reviewed full sha>`. If the head moved after GO, run `~/.claude/skills/run-wave/scripts/delta-kind.sh <reviewed> <head> origin/accelerate`. Merge only on `trivial`; `code` means another delta review. Never chain delta-kind and merge in one command.
- GitHub shows CONFLICTING whenever two PRs both append to `.claude/skills/sdk/SKILL.md`. Merge `origin/accelerate` into the branch locally, add a blank line between the appended notes, push, run delta-kind (it says `trivial`), then merge.
- Each PR gets its own worktree, scratch folder and test DB: `source <scratch>/w3c/testdb.sh <worktree> model_router_test_<slug>`. **Never commit `acceleration/internal/config/testing.yaml`.**
- Severity: Blocker and Should fix block the merge and are fixed in the PR. Nits and Questions become Linear tickets, **one ticket per problem**: team AI, project "AI - Part two", parent AI-816, assignee Kanat. Put "what to do" in the description and "what was done" in a comment.
- After 3 consecutive NO-GOs on one PR, stop and bring Kanat a decision (split or narrow the PR).
- Decide in-wave items yourself (take your recommended option, then tell Kanat). Hard gates wait for Kanat.

## 3. Rules from Kanat (still in force)

- **Never** run a data move (`router plugins migrate --apply`, any backfill) or switch connectors/plugins on a shared environment without Kanat's explicit OK. Dry runs on shared envs also need his go. «миграцию и переход на connectors сам не делай - будем делать в ручном режиме со мной».
- Local DB and router actions need no approval; take a `pg_dump` first if the action is hard to undo.
- Outward-facing actions need Kanat's OK: Slack posts outside #kanat-test, Stream app or Slack app settings, staging deploys, chat repo infra.
- Never print secrets (tokens, client secrets, KEK, `.env` values, PATs, cookies, auth headers). Use sha256 fingerprints. Agents must not read cookies or session tokens. Agents must never kill processes broadly (no `pkill -f vite`).
- Evidence only: cite `file:line`, a command or a commit for every claim; mark anything you have not checked `unverified`.
- Volt work follows the repo's Design Standards (memory `volt-design-standards.md`).
- Router behaviour PRs ship with their first production caller; expand-only migrations, numbered after the newest on base.
- Token economy: short prompts, report budgets, fresh fixers, delta review by the same reviewer while its context is under ~200k.

## 4. Current state (verified 2026-10-10)

**Staging router:** `accelerate-v0.6.34` = `e464eeeb`, deployed 05:32Z, goose at `20261016120000`. Health check after deploy: 0 restarts, 0 ERROR, no 5xx, `/health` 200, 150 s recheck clean. `ROUTER_CONNECTORS_ENABLED` is on. `DASHBOARD_BASE_URL` = the Vercel preview URL. Rollback target is v0.6.33; a binary rollback is safe, because v0.6.33 was tested against a DB migrated to 20261016120000. The exact deploy, verify and rollback commands are in memory `staging-migration-order.md`; always run `launch inspect` first.

**Staging gateway:** image `v0.1.4`, chat#18368/#18369 (helm rev 19), plus the test-only follow-up chat#18370. It forwards `X-Stream-User-Id` for server tokens (a user token may only name itself, otherwise 403) and allowlists the events and mcp-events routes. The image is built locally; the recipe is in the memory file, because `build-accelerate.yml` is not on the default branch.

**Dashboard preview** (staging router behind it): https://volt-dashboard-git-ai-team-agent-dashboard-getstreamio.vercel.app/organization/1181507/1257545/agents/
- Library › Connectors (tabs Connectors and Connections). Agent › Tools › Connectors. «Connected apps» is hidden behind `SHOW_PLUGINS = false`.
- Pushing to `ai-team/agent-dashboard` does NOT deploy the preview. Run `gh workflow run deploy-vercel.yml --ref ai-team/agent-dashboard`.

**Staging test app 1257545** (org 1181507, the Stream app «Video Demo App»):
- Agent `e2e-connectors` (a54fb800…) binds Linear (session), Slack bot (fixed, connection `e2e-connectors-slackbot`) and the demo `deepwiki`.
- **Demo custom connector for developers:** `custom_demo_deepwiki` (https://mcp.deepwiki.com/mcp), connection `demo-deepwiki-public` (22cb6c2f…), fake api key `X-Demo-Key: public-no-auth` (AI-1059: no "no auth" scheme yet). Working prompt: «Use DeepWiki: list the documentation topics of the GitHub repo GetStream/Vision-Agents (read_wiki_structure), then tell me the first five.» Keep it.
- Kanat's GitHub PAT connection was deleted. The PAT itself was not revoked at GitHub; that is Kanat's call.
- A pending `e2e-connectors-slack` connection exists, because Slack user OAuth cannot start: there is no Stream operator Slack client on staging (F87, Kanat's decision).
- **Stream `message.new` hook** of app 1257545 points to `https://accelerate.gcp.stream-io-api.com/v1/chat/hooks/stream/1257545`. The pre-change copy, with secrets redacted, is `_connectors_handoff_docs/migrate/hooks-before.json`. The diagnosis is in `slackbot-drop.md`. The call hooks still point to ngrok, plus someone else's `eec4-…` hook; do not touch them.
- **Slack bot app** (`accelerate-bot-test`, provider app A0C7N6LNZMH): the Event Subscriptions Request URL points to staging, and the staging redirect URI was added. The old local ngrok URL was `https://carol-elliptic-uncloak.ngrok-free.dev/v1/connectors/events/slack_bot/A0C7N6LNZMH`. The staging E2E passed end to end: a hand-typed mention in #kanat-test got a reply in the thread at 03:29Z.

**Local stack** (for UI E2E): compose project `vision-agents` (router, postgres, redis) is running. Its router is built from 599298c6 and is stale: rebuild it before testing with `docker compose -f compose.yaml -f ../volt-dashboard/docs/local-agents/compose.volt.yaml up -d --build router`. The dashboard dev server runs on https://local.getstream.io:3000 from worktree `<scratch>/volt/wt-int`, started detached; its PID file is `<scratch>/volt/dev-int.pid`. ngrok needs `NGROK_AUTHTOKEN` from `.env`. The local slack_bot inbound no longer works, because both the Slack events URL and the Stream hook moved to staging (Kanat chose this).

**Plugins → connectors:** there is no plugin data to move. Staging has 0 plugin clients and 0 plugin connections, and the only plugin entry (config 1731296 `loubot_grader` salesforce) is skipped by the dry run. Kanat decided on a **two-phase removal** (memory `plugins-two-phase-removal.md`):
- **Phase 1 is done:** #872 is deployed in v0.6.34. It adds OpenAPI `deprecated`, WARN logs `msg="deprecated plugin path used"` with `path=… via=plugins|mcp_servers unbound=<n>`, and a Go SDK warning.
- **Phase 2**, deletion, earliest ~2026-10-24, only with Kanat's go and evidence: 0 real plugin WARNs over the period (exclude `via=mcp_servers` and `unbound=0`) and healthy connectors E2E.
- The removal plan is in `_connectors_handoff_docs/migrate/plugin-removal-inventory.md`.
- Caveat: `mcp_servers` with a login runs on the plugin code, so it needs a replacement before deletion.
- At deploy the WARN count was 0.

## 5. Wave history and what shipped

Waves 3b–7 shipped the connectors core. Their ledgers are in `_connectors_handoff_docs/waves/`. The volt connectors UI phases A–I were volt PRs #994–#1001 and #1003, integrated by #1002 (merged 53c429906e). **Wave 8 (done, v0.6.34):**

| PR | Ticket | What |
|---|---|---|
| #871 | AI-1053, F52, F67 | `Connector.channel`; the delete log names the dropped fingerprint; the `skip_if_present` skip is logged at Info |
| #873 | AI-958 | proxy cap of 60 calls/min per app per connector (`ROUTER_CONNECTORS_PROXY_CALLS_PER_MINUTE`); refuses encoded `/` `\` `..`; slack_bot revision 5 with `api_base` |
| #874 | AI-1052 | `last_validation` (side table, migration 20261016120000); on validate a 4xx other than 429 moves a static connection to `needs_reauthorization`; the provider text is stripped of credentials and capped at 1 KiB |
| #875 | AI-1049, AI-1048 | one live owner config per channel or event connection; test copies (`tags.draft_of`) excluded; 409 `errChannelConnectionTaken` |
| #877 | — | saving an OAuth client points the Stream message hook; startup WARN when a hook misses this router; Info logs for silent inbound drops; event_hooks round-trip as raw JSON (times as RFC 3339) |
| #872 | — | plugins deprecated, phase 1 |
| #876 | — | skills `.claude/skills/connectors` and `add-connector` for other developers |

## 6. Where the internal docs are

- **Durable copy, made for this handoff:** `~/Projects/stream/Vision-Agents/_connectors_handoff_docs/`
  - `waves/`: wave ledgers and briefs (`8.md` is the latest; `3d-brief.md` + `wave3-common.md` are the common brief every agent reads first; `8-brief.md`).
  - `volt/`: `plan.md`, `brief.md` (volt agent rules, design standards), `phase-h.md`, phase findings, local E2E reports `e2e-d*.md` (findings F47–F74).
  - `migrate/`: `plugin-removal-inventory.md`, `e2e-staging.md` (staging E2E, F80–F89), `slackbot-drop.md` (the hook incident).
  - `wave8-findings/`: per-PR findings lists of wave 8.
  - `e2e-local/`: the earlier local connector E2E reports and findings.
- **Original scratchpad, ephemeral, may be gone:** `/private/tmp/claude-504/-Users-kanat-Projects-stream-Vision-Agents/a0abf0a3-74bd-4641-9e95-9b4aed318e30/scratchpad/`. It holds the scripts too: `w3c/testdb.sh`, `w8/baseline.sh`, deploy and verify scripts in `deploy-v0.6.34/` and `cors-fix/`, `canary/psql-pod.yaml` (a read-only psql pod). If it is gone, recreate `testdb.sh` from the description in the run-wave mechanic template and in memory `parallel-agents-test-db-isolation.md`.
- **Persistent memory** (auto-loaded): `~/.claude/projects/-Users-kanat-Projects-stream-Vision-Agents/memory/`, with the index in `MEMORY.md`. The key files:
  - `staging-migration-order.md`: every staging deploy, verify and rollback command, the gateway image recipe, history;
  - `plugins-two-phase-removal.md`;
  - `wave-workflow.md`, `token-efficient-waves.md`, `pr-review-gate-before-merge.md`;
  - `local-db-no-approval.md`, `one-ticket-per-problem.md`, `volt-design-standards.md`, `accelerate-staging-only.md`.
- **Plan doc:** `acceleration/docs/connectors/subtasks.md` on `origin/connectors/planning`.
- **Developer-facing:** the `connectors` and `add-connector` skills in the repo.

## 7. What to do next (proposed to Kanat, not yet approved)

Kanat was asked «Запускать фазу J и планировать волну 9?» and has not answered yet. Wait for his answer.

1. **Volt phase J** (one PR off `origin/ai-team/agent-dashboard`). Kanat's design lead said «Feel free to change the UX/UI or section position the way which you think fits the best». Scope:
   - move Connectors to the top of the agent's Tools tab;
   - read `Connection.last_validation` from the API (the dashboard keeps «Last check» only in cache today);
   - use `Connector.channel` in the agent delete dialog (AI-1069);
   - make Duplicate and Save as new drop or swap a slack_bot binding, which now hits 409 (AI-1083, Medium);
   - AI-1057 (0 tools, comma-separated names), AI-1058 (silent consent failure), AI-1062 (raw id flash).
   Process as before: author, independent review, browser check on :3000 against a **rebuilt** local router, merge, then `gh workflow run deploy-vercel.yml --ref ai-team/agent-dashboard`.
2. **Router wave 9.** First **AI-1078 (High, security)**: `POST …/validate` still returns the provider's error text with the credential; apply `storedError` stripping to the answer. Then AI-1051 (bindings left after a forced connector delete), F55 (Linear DCR registers a new client per consent), AI-1050 (shared Slack tokens; verify first). Optional: AI-1067, AI-1079–AI-1081. Plan with `plan-waves`: newest migration on base is now `20261016120000`.
3. **Gates waiting for Kanat:**
   - F87: a Stream operator Slack client on staging (a secret in the staging secrets plus a chart change);
   - AI-946: a sandbox Stream app with an SQS hook, to verify the hook round-trip live (outward-facing);
   - AI-1076: a per-connection share of the proxy cap, which Kanat will think about next week;
   - AI-1082 (MCP-Protocol-Version 400) and AI-1084 (a test-copy field): Questions;
   - plugins phase 2 (~2026-10-24).
4. **Known base failures** (do not diagnose): CI js, mypy and python-unit have been red since colleague commit 9f629685. Order-dependent or flaky integration suites: TestSIPTrunksSuite, TestTurnRecordingSuite, TestDataMoveSuite, TestKnowledgeUrlsSuite, TestConnectionEventsSuite, TestSessionUpdateSuite/TestAUserRenamesASessionThatEnded, TestQuotaSuite, TestServerSuite/TestProvidersReportLiveHealth, TestSTSRouterSuite/TestInterruptMutesTheReplyBeforeTheProviderHearsOfIt. gofmt flags `streams_test.go:382`. The full list is in `waves/8.md`.

## 8. Open Linear tickets created in this session (all under AI-816)

- AI-1054, AI-1055 (gateway allowlist path, handshake verify token)
- AI-1056–AI-1063 (staging E2E F81–F89)
- AI-1064–AI-1069 (#871)
- AI-1070–AI-1076 (#873)
- AI-1077 (inbound queue rework)
- AI-1078–AI-1082 (#874)
- AI-1083–AI-1086 (#875)
- AI-1087–AI-1093 (#877, #872)

Closed this session: AI-958, AI-1048, AI-1049, AI-1052, AI-1053.
