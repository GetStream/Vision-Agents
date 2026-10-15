# Connectors (AI-816) handoff, 2026-10-15 15:20Z

> **Do not commit this file or `_connectors_handoff_docs/`.** The repo is public. They are working notes. Keep GCP project, cluster and secret names out of anything you commit; those details live only in the memory file `staging-migration-order.md`.

You are taking over from the orchestrator agent that ran the AI-816 connectors work for Kanat (kanat.kiialbaev@getstream.io). Read this whole file before you act. Reply to Kanat in **Russian**, keep English code names untranslated, use they/them for people, and be brief.

## 1. What the project is

**Connectors** replace the old **plugins** (Connected apps). A **connector** is a service an agent can use: a built-in (Slack, Slack bot, GitHub, Linear and 14 more, defined as manifest YAML in the router) or a custom MCP server one app adds. A **connection** is an account connected to a connector, owned by the app or by one end user. A **binding** gives an agent config a connector's tools: `fixed` (one app connection) or `session` (each user's own account, consent in the Playground card). **Channel connectors** (slack_bot, linq, telnyx, whatsapp) also receive inbound messages.

Inbound flow today: provider event → router `/v1/connectors/events/{connector}/{provider_app}` → the router writes the message into a thread channel in the app's **Stream Chat** → Stream fires `message.new` to the router's hook `/v1/chat/hooks/stream[/<app id>]` → the agent replies. The planned rework (router hands the message to the agent directly, router-side queue, `message.new` becomes the ACK) is **AI-1077**, assigned to Kanat.

Repos and branches:

| Repo | Path | Branch | What |
|---|---|---|---|
| GetStream/Vision-Agents (public) | `~/Projects/stream/Vision-Agents` | `accelerate` (base for all router PRs) | Go router in `acceleration/`, SDKs, skills |
| volt-dashboard (private) | `~/Projects/stream/volt-dashboard` | `main` (PRs target `main`; `main` deploys production) | The product dashboard with the Agents section and the connectors UI |
| GetStream/chat | `~/Projects/stream/chat` | `thierry/accelerate-gke` | Staging gateway proxy (`projects/stream-accelerate/proxy.go`), helm chart, staging values, `rocky` CLI |

Branch naming: router PRs use `connectors/<slug>` off `accelerate`. The volt integration branch `connectors/agents-ui` is merged (#1002) and gone from origin; volt PRs now branch off and target `main` (Colin, #video-ai 2026-10-15: everything from `ai-team/agent-dashboard` is in `main`, e.g. squash `e4a0a3962`). A merge to `main` deploys production, so it needs Kanat's go.

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
- Before tagging, compare every new migration file with the live goose version. A file numbered below that version must not ship: goose has no out-of-order, and the router crash-loops. Learned 2026-10-12: `c891381c` carried `20261015120000` under staging's `20261016120000`.
- Token economy: short prompts, report budgets, fresh fixers, delta review by the same reviewer while its context is under ~200k.

## 4. Current state (updated 2026-10-15 15:20Z; older notes below are history)

**Start here (2026-10-15).**

- **Staging router.** v0.6.51 = `5602ab70` was deployed at 2026-10-15T13:48Z and migrated goose to `20261031130000`; 0 restarts, no 5xx. Since then `accelerate-v0.6.52` = `35e23fdc` (tip of `accelerate`) was tagged at 14:45Z by another session, and the router pods restarted at ~15:08Z and ~15:16Z (`goose: no migrations to run. current version: 20261031130000`). Which version runs now is unverified: run `launch inspect`. Earlier colleague tags: v0.6.49 `446e7256`, v0.6.50 `1ec4fc49` (Nash, 2026-10-15 04:39Z). Rollback target for v0.6.51 is v0.6.50 (binary only, the migration is expand-only).
- **Google Workspace MCP 403 (Drive, Gmail, Calendar): solved, no code bug.** Every `tools/call` returned "The caller does not have permission" because two Google gates were closed: (1) the staging operator Google OAuth client must be flagged "Use this client for an AI-powered agent" (Kanat ticked it 2026-10-14 ~20:10Z); (2) its GCP project must be enrolled in the Workspace Developer Preview (applied 2026-10-14 ~23:43Z, approved by Google's email 2026-10-15 01:06:47Z). Evidence: Google's `serviceruntime.googleapis.com/api/request_count` on that project shows Drive 403s on 2026-10-14 19:44–20:21Z, then Gmail/Calendar 200s from 05:05Z (app 1731781) and Drive `SearchFiles` 200 at 15:15Z. Neither gate is visible or settable from the CLI. Project ids, client id and the email: memory `reference_google_workspace_mcp_preview_gate.md` (chat-agent project).
- **Drive verified end to end 2026-10-15 15:14Z** on staging app 1257545: connection `a402982f6530e16e7ce33592b4fc9fe0`, throwaway agent config `kanat-drive-check` (deleted 16:4xZ), session `01a14020-d681-7a94-a723-608c219503c1`; `connector_invocations` row `a1923b10…` `search_files`, no error, 930 ms.
- **Tool pin gotcha.** A connection first used before a provider changed a tool's schema is left out of sessions: `connector_unavailable reason=tool_unavailable`, log "a tool a pattern matches is not offered: the connection no longer lists it with the schema it was pinned at" (`acceleration/internal/session/connector_tools.go:444-473`). `POST /connections/{id}/validate` does not clear it; only a reconnect (new consent, new `connected_at`) re-pins. Hit on `a402982f` (pinned 10-13); Kanat reconnected it 15:12:49Z (revision 6). A CLI reconnect works: `POST /v1/agents/connections/{id}/authorizations`, open `launch_url` in Chrome, POST `{handoff_token}` to the launch path from that page (Origin must be the router), follow `authorization_url`.
- **Tool pin drift fixed (AI-1302, Done), on staging:** #939 in v0.6.55 (`de67e3d7`, 20:09Z), #946 in v0.6.56 (`d65e75dd`, deployed 20:45Z, goose `20261101130000`, healthy at t0/150 s/6.5 min; rollback v0.6.55 binary-only). #939 AI-1304 `9a74ce52`: validate answers `needs_reapproval` + `changed_tools`. #946 AI-1305 `7da78101`: `POST /v1/agents/connections/{id}/tools/approve` re-pins at the listed digest (server-side only, compare-and-set, audit action `tool_reapproved`, migration `20261101130000` re-adds the CHECK `NOT VALID`). Follow-ups AI-1332, AI-1333, AI-1347..AI-1350. Ledger `~/.claude/ship-task/Vision-Agents/AI-1302/ledger.md`.
- **All four Google connectors verified 2026-10-15** on app 1257545 (drive `search_files`, docs `read_doc`, calendar `list_events`, gmail `search_threads`; no errors). AI-1176 closed. The consent screen is Internal (only getstream.io accounts) and branded rocky-ui: AI-1172. No production Accelerate environment exists. Test agent configs deleted.
- **Yun's Slack thread** (app 1731750, agent `hackathon_superman`): the reply asking Yun to retry `search_files` (and reconnect if Drive is missing) is drafted and **not posted**; posting is Kanat's call.
- **Connector error logging (AI-1241, Done).** Four PRs on `accelerate`, all on staging from v0.6.51: #927 AI-1242 `11102295` (log failed connector calls), #929 AI-1243 `11fae4b1` (provider host, status, 1 KiB redacted body, `WWW-Authenticate`/`Retry-After`), #930 AI-1244 `8666b26d` (migration `20261031130000`: `connector_invocations.provider_status`, `error_message`, exposed on `GET /v1/agents/connections/{id}/invocations`; router rate-limiter holds store "rate limited by the router: retry after <n>s"), #933 AI-1245 `5602ab70` (consent-denied, refresh-failed and REST proxy non-2xx logs; the proxy no longer forwards the Stream server-token headers to providers, Kanat-approved). The model-facing tool result and every `error_type` are unchanged. Time column of `connector_invocations` is `started_at`. Ledger: `~/.claude/ship-task/Vision-Agents/connector-error-logging/ledger.md`. Follow-ups: AI-1248..1252, AI-1260..1263, AI-1270..1277, AI-1281, AI-1282; AI-1273 (High: redaction misses JSON-escaped `\/` and `&`, which can now reach the API) is next; AI-1270 (incognito rows keep error_message) and AI-1272 (Rust/PHP regen) wait for Kanat; plugins path is AI-1246. #928 (`d1ac7dd7`) fixed the red `go` CI on accelerate (AI-1258, AI-1259).
- Older "Start here" notes from 2026-10-13 follow; their staging version is stale.

**2026-10-13 notes.** Staging was `accelerate-v0.6.35` = `8160c4cd`. [#905](https://github.com/GetStream/Vision-Agents/pull/905) merged to `accelerate` at 2026-10-13T18:20:23Z as `cfd530953d` (`Shopify storefront needs no login`). It is not on staging. Kanat asked to merge it and did not ask to deploy. A staging deploy still needs Kanat's OK (section 3).

#905: `providers/shopify.yaml` revision 2, `schemes: [none]`, description «A shop's policies and FAQs.», revision 1 marked broken (`asked for an OAuth login; the storefront MCP at /api/mcp needs none`). `/api/mcp` is `search_shop_policies_and_faqs` only. Checked 18:15Z: `POST https://stream-agents-e2e.myshopify.com/api/mcp` `tools/list` returned that one tool with no `Authorization` and with a bearer the shop was not issued (both 200; unauthenticated request `eec350f1-1bc9-41ec-aaf4-bf792611e685-1791915330`). Sources in the yaml: shopify.dev storefront-mcp and policy-and-faqs-tool, opened 2026-10-13. A plugin grant is `oauth2_code` and does not move onto this revision. A catalog or merchant connector (`/api/ucp/mcp`) is a separate ticket, not filed. Written in the parent chat, no review round: Kanat asked to finish the merge and to cut token use.

**Wave 9 is done and on staging.** `accelerate-v0.6.35` = `8160c4cd` (router and gateway), deployed 2026-10-13T00:12:48Z. Pre-tag goose gate passed (live was `20261016120000`). Goose is now `20261018120000`: 17 index swap (`20261017120000`), 18 custom_models (`20261018120000`, the #897 move of `75214efe`). t0 and the 00:32Z recheck: 0 restarts, 0 ERROR, 0 5xx, `/health` 200. No rollback. Rollback target is v0.6.34 (it starts on a DB at 18: no migrations to run). `tool_patterns` is in the spec at this commit. The live process does not serve `/openapi` (404), so the live spec is unverified. Ships: #891 #892 #893 #895 #896, #897, colleague #887 #898 #899 (JS SDK reads `connectors:`), and `75214efe` custom models. CLI beta `v2026-10-13-eab135f` is released. Docs getstream.io#651 is merged into `agents` (`f2211fa4`). Syntax and rulings: `_connectors_handoff_docs/waves/9-current.md`. Ledger: `waves/9.md` (do not edit). Exit tickets AI-1105..AI-1145 under AI-816; notable ids in section 8.

**Wave 10 (Volt phase J) is done** on `ai-team/agent-dashboard` at `dce53973`. No router tag. volt-dashboard #1013 #1015 #1016 #1017 #1014 merged. Vercel run 38348753114. Smoke pass 05:08Z on staging app 1257545 (console errors unverified; see `waves/10-current.md`). Deferred to wave 11: `Connection.last_validation` (F70), F73. Ledger: `waves/10.md` (do not edit). Exit tickets AI-1146..AI-1162.

**Staging router:** `accelerate-v0.6.35` = `8160c4cd`, deployed 2026-10-13T00:12:48Z, goose `20261018120000`. Health: t0 and 00:32Z recheck, 0 restarts, 0 ERROR, 0 5xx, `/health` 200. `ROUTER_CONNECTORS_ENABLED` is on. `DASHBOARD_BASE_URL` = the Vercel preview URL. Rollback target is v0.6.34 (starts on DB at 18, no migrations to run). A pattern binding rolled back to v0.6.34 grants nothing (fail closed). Commands: memory `staging-migration-order.md`; always run `launch inspect` first. Before the next tag, compare new migration files with the live goose version (no file below it).

**Staging gateway:** redeployed with the router at 2026-10-13T00:12:48Z. Image tag not re-read this sync (last recorded `v0.1.4`, chat#18368/#18369, helm rev 19, plus chat#18370). It forwards `X-Stream-User-Id` for server tokens (a user token may only name itself, otherwise 403) and allowlists the events and mcp-events routes. The image is built locally; the recipe is in the memory file, because `build-accelerate.yml` is not on the default branch.

**Dashboard preview** (staging router behind it): https://volt-dashboard-git-ai-team-agent-dashboard-getstreamio.vercel.app/organization/1181507/1257545/agents/
- Library › Connectors (tabs Connectors and Connections). Agent › Tools › Connectors. «Connected apps» is hidden behind `SHOW_PLUGINS = false`.
- Latest deploy: Vercel run 38348753114 at `dce53973` (alias last-modified 04:58:31Z, deploy url `volt-dashboard-czsf2gzxm`, alias step did not skip). Smoke pass 05:08Z on app 1257545. Pushing the branch still does not deploy. The command is `gh workflow run deploy-vercel.yml --ref ai-team/agent-dashboard`.

**Staging test app 1257545** (org 1181507, the Stream app «Video Demo App»):
- Agent `e2e-connectors` (a54fb800…) binds Linear (session), Slack bot (fixed, connection `e2e-connectors-slackbot`) and the demo `deepwiki`.
- **Demo custom connector for developers:** `custom_demo_deepwiki` (https://mcp.deepwiki.com/mcp), connection `demo-deepwiki-public` (22cb6c2f…), fake api key `X-Demo-Key: public-no-auth` (AI-1059: no "no auth" scheme yet). Working prompt: «Use DeepWiki: list the documentation topics of the GitHub repo GetStream/Vision-Agents (read_wiki_structure), then tell me the first five.» Keep it.
- Kanat's earlier GitHub PAT connection was deleted. The PAT itself was not revoked at GitHub; that is Kanat's call. B3 (9.md log 03:06Z): `e2e-matrix-github-pat` on staging, `get_me` passed.
- F87 is done. Staging Slack uses the existing app A0C7L86EEAX. Connected row `838b06742f5aac505effc2ccf0b50ef6` (prior session, not rechecked 18:20Z). The unused app A0C8ZMKLXBN was not installed.
- **Stream `message.new` hook** of app 1257545 points to `https://accelerate.gcp.stream-io-api.com/v1/chat/hooks/stream/1257545`. The pre-change copy, with secrets redacted, is `_connectors_handoff_docs/migrate/hooks-before.json`. The diagnosis is in `slackbot-drop.md`. The call hooks still point to ngrok, plus someone else's `eec4-…` hook; do not touch them.
- **Slack bot app** (`accelerate-bot-test`, provider app A0C7N6LNZMH): the Event Subscriptions Request URL points to staging, and the staging redirect URI was added. The old local ngrok URL was `https://carol-elliptic-uncloak.ngrok-free.dev/v1/connectors/events/slack_bot/A0C7N6LNZMH`. The staging E2E passed end to end: a hand-typed mention in #kanat-test got a reply in the thread at 03:29Z.

**Local stack** (for UI E2E): compose project `vision-agents` (router, postgres, redis) is running. Router rebuilt 2026-10-13 03:05Z from worktree `<scratchpad>/w10/va-accel` at origin/accelerate `2ab444fd` (includes `8160c4cd`): goose `20261018120000`, `/health` 200. Dump before the rebuild: `<scratchpad>/w10/local-db-before.sql`. Lesson: socket CORS (`ROUTER_CORS_ORIGINS`) allows only `https://local.getstream.io:3011`; playground checks use that port, one agent at a time. The dashboard dev server on https://local.getstream.io:3000 is worktree `<scratchpad>/volt/wt-int` (PID file `<scratchpad>/volt/dev-int.pid`); it is behind base (53c429906e). Do not move or stop it. ngrok needs `NGROK_AUTHTOKEN` from `.env`. The local slack_bot inbound no longer works, because both the Slack events URL and the Stream hook moved to staging (Kanat chose this). A local DB that recorded `20261015120000` makes `router plugins migrate` refuse the #897 file: delete that `goose_db_version` row. A branch that still carries `20261015120000` fails on a DB migrated by #897 until it merges `accelerate`.

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

**Wave 9 (done, v0.6.35, hackathon 2026-10-14).** On tag `8160c4cd`: #891 AI-1078 (`06b7bce4`), #892 (`dff5f98e`), #893 `tool_patterns` (`5857314d`), #895 Go SDK (`29a014f0`), #896 Python SDK (`c891381c`), #897 migration move (squash `8160c4cd`). Also on the tag: colleague #887 #898 #899 (JS SDK reads `connectors:`) and custom models `75214efe` (goose 18). GetStream/cli#47 is on `beta` (`eab135fa`); release `v2026-10-13-eab135f` is out. Docs getstream.io#651 is merged into `agents` (`f2211fa4`; readers: CLI, Go, Python, JS; Ruby/PHP/Rust/.NET refuse). Detail: `waves/9-current.md`. Exit tickets: section 8. Router SHAs are from the ledger log 00:42Z and the deploy report, not a fresh `gh pr view` this sync.

**Wave 10 (done, Volt phase J, no router tag).** volt-dashboard #1013 (`55b3afc9`, AI-1062), #1015 (`07e6a4e3`, AI-1056), #1016 (`0b9fc9b0`, AI-1083, AI-1069), #1017 (`ef80bae3`, AI-1120, AI-1057), #1014 (`dce53973`, AI-1058). Tip `dce53973`. Vercel run 38348753114. Smoke pass on app 1257545. Deferred: F70 `Connection.last_validation` and F73, both wave 11. Detail: `waves/10-current.md`. Rulings J1-J9 are closed.

## 6. Where the internal docs are

- **Durable copy, made for this handoff:** `acceleration/docs/connectors/_connectors_handoff_docs/` (this file sits beside that folder).
  - `waves/`: ledgers and briefs. `9.md` and `10.md` are the wave ledgers; the orchestrator owns them, so a handoff sync links to them and does not edit them. `9-current.md` is the wave 9 handoff detail (yaml syntax, `tool_patterns`, PR SHAs, staging v0.6.35). `10-current.md` is the wave 10 handoff detail (phase J rulings, volt PR SHAs, Vercel deploy, smoke). `9-brief.md` and `10-brief.md` are the agent briefs. `3d-brief.md` + `wave3-common.md` are the common brief every router agent reads first (`wave3-common.md` does not apply to volt). `8.md` is the wave before 9.
  - `volt/`: `plan.md`, `brief.md` (volt agent rules, design standards), `phase-h.md`, phase findings, local E2E reports `e2e-d*.md` (findings F47–F74).
  - `migrate/`: `plugin-removal-inventory.md`, `e2e-staging.md` (staging E2E, F80–F89), `slackbot-drop.md` (the hook incident), `operator-clients.md` (vendor OAuth client prep for the hackathon pass; no secret values).
  - `wave8-findings/`: per-PR findings lists of wave 8.
  - `e2e-local/`: the earlier local connector E2E reports and findings.
- **Original scratchpad, ephemeral, may be gone:** `/private/tmp/claude-504/-Users-kanat-Projects-stream-Vision-Agents/a0abf0a3-74bd-4641-9e95-9b4aed318e30/scratchpad/`. It holds the scripts too: `w3c/testdb.sh`, `w8/baseline.sh`, deploy and verify scripts in `deploy-v0.6.34/` and `cors-fix/`, `canary/psql-pod.yaml` (a read-only psql pod). If it is gone, recreate `testdb.sh` from the description in the run-wave mechanic template and in memory `parallel-agents-test-db-isolation.md`.
- **Persistent memory** (auto-loaded): `~/.claude/projects/-Users-kanat-Projects-stream-Vision-Agents/memory/`, with the index in `MEMORY.md`. The key files:
  - `staging-migration-order.md`: every staging deploy, verify and rollback command, the gateway image recipe, history;
  - `plugins-two-phase-removal.md`;
  - `wave-workflow.md`, `token-efficient-waves.md`, `pr-review-gate-before-merge.md`;
  - `local-db-no-approval.md`, `one-ticket-per-problem.md`, `volt-design-standards.md`, `accelerate-staging-only.md`.
- **Plan doc:** `acceleration/docs/connectors/subtasks.md` on `origin/connectors/planning`.
- **Developer feedback (Notion):** [WWSD platform feedback](https://app.notion.com/p/stream-wiki/WWSD-platform-feedback-3fa6a5d7f9f6816ea917cb967a124566) (WWSD = "What Would Stream Do"). Our developers wrote down the issues they hit while working with the platform. Read it before you plan connectors work for developers.
- **Developer-facing:** the `connectors` and `add-connector` skills in the repo.

## 7. What to do next

Wave 10 is done (`dce53973` on `ai-team/agent-dashboard`). Staging router is still `accelerate-v0.6.35` = `8160c4cd`. `accelerate` tip includes #905 (`cfd530953d`), not deployed. Do not edit `waves/9.md` or `waves/10.md`. The 03:05Z "6 vendor tabs waiting" note is stale; do not repeat those logins.

1. **Deploy #905, then connect Shopify.** Ask Kanat before the deploy. After it: connection input `stream-agents-e2e.myshopify.com`, scheme `none` (Begin is Done, no OAuth). Pending connection from the prior session: `fefb8ee97949f6359d69f714f66ceb5b` (not rechecked 18:20Z). Dev store admin `https://admin.shopify.com/store/stream-agents-e2e`. App "Stream Agents (staging)" is installed on that one store only.
2. **B3 tool calls, not done.** Prior session (not rechecked 18:20Z) left these Connected on app 1257545, Validate Works, no agent tool call yet. Matrix draft `<scratchpad>/b3-e2e.md` may be gone with the scratchpad. One read-only call each, plus whether Cal.com refreshed a token that expired 12:19. Do not post to Slack.
   - calcom `512bf6c38edd994ab1fb4b8df31a2cfa` (66 tools)
   - sentry `50c39b62f0539add36058be136f553dd` (15 tools; consent granted writes, dashboard said "Scopes: None")
   - calendly `a4e7d4e984d0e3d14627381c6d34e7bf` (36 tools)
   - google_calendar `9ec9ea4d6db863cb1c5dd8a49f23aa36` (9)
   - google_drive `a402982f6530e16e7ce33592b4fc9fe0` (8)
   - google_docs `67a7d931a2dedc70f2fc26ab585a705f` (2)
   - gmail `4cbf237edc023878aad4f803145e384b` (23)
   - hubspot `96e9d21f2002274756cfb9fde9c25de2` (29, account 343781562)
   - slack `838b06742f5aac505effc2ccf0b50ef6` (26, T02RM6X6B:U034NG4FPNG). Staging uses existing app A0C7L86EEAX, not the unused A0C8ZMKLXBN. Events URL points at staging.
   - github `0877f93e532e6c0e214bc2af6d60a105` (49). Client id `Ov23lieXMgFV24pNfGdU` (the stored id had been misread).
   - salesforce `0b1dd95f75f1bc13c86a74c6be82f426` (11). Developer Edition org `00Dbm000014Ictj`, user `kanat.kiialbaev.efd2a8d76d29@agentforce.com`. Idle refresh TTL is locked at 30 days. Old Starter org power-efficiency-7834 is superseded. Chrome default context may still be that old org.
   - Linear `list_teams` and GitHub PAT `e2e-matrix-github-pat` / `get_me` already passed (9.md).
3. **Blocked, do not retry signup.** Gong: paid seat plus Technical Administrator; automatic vs manual registration is permanent. Instacart: production Developer Platform key, 30-40 days for a new one; a dev key does not auth `mcp.instacart.com`.
4. **Do not post** `<scratchpad>/thierry-reply.md`. Kanat posts it.
5. **Tickets not filed.** UCP or Admin Shopify connector. Sentry tool-count vs consent scopes, and the dashboard "Scopes: None". Google consent branded `rocky-ui`. HubSpot unverified-app warning. Salesforce 30-day idle refresh TTL. GitHub OCR credential-copy incident.
6. **Cleanup, not done.** User-owned Calendly connections `84be36086b96c7110d063518ec80ccd7` (Connected) and `8a5e517efe4199b95209f0dd292c8996` (Pending). Unused Slack app A0C8ZMKLXBN.
7. **Tokens.** Kanat said consumption skyrocketed (2026-10-13 18:12Z). Summarize does not shrink the rules prefix or subagent bills. Next session: one worker, no Opus review loop unless Kanat asks, drop unused workspace roots.
8. **Volt phase J is done** (plan track D; five PRs, ruling J1). Merged #1013 #1015 #1016 #1017 #1014. Shipped: Connectors on top of Agent > Tools; `tool_patterns` preserved (AI-1120; saving a pattern binding keeps `tool_patterns` and sends `tools: []`); comma/whitespace tool names (AI-1057); consent failure keeps the dialog open (AI-1058); skeleton rows until the name is read (AI-1062); Duplicate and Save as new drop a channel binding the router would 409 (AI-1083, J4); delete dialog names only an exclusive channel connection (AI-1069); `connector_unavailable` notice (AI-1056). Kanat: Q1 yes, Q2 block (no `*` default), Q3 defer, Q4 drop; Q5 wording taken as recommended. Detail and closed rulings: `waves/10-current.md`.
   - **Deferred to wave 11:** read `Connection.last_validation` (F70; "Last check" is still cache-only). F73 (info hint rendered as `role=alert`).
9. **Still waiting, not this sync.** AI-946 (sandbox Stream app with an SQS hook), AI-1076 (per-connection proxy cap), AI-1082, AI-1084, plugins phase 2 (~2026-10-24). Deferred off wave 9: AI-1051, F55, AI-1050, AI-1067, AI-1079-AI-1081.
10. **Known base failures** (do not diagnose): the list in `waves/9.md`, except the 3 ca8bcd5d session tests. #892 fixed those (`TestASharedConversationUsesTheAppsConnectionsOnly`, `TestNoSelectionImpliesNothingForAnUnverifiedCallerOrASharedConversation`, `TestOnlyAnOptionalBindingOfTheCallersOwnWaitsForALogin`). `9.md` still names them in its baseline snapshot; ignore that. gofmt parse error is `streams_test.go:417` (the wave 8 note said `:382`). CI js, mypy and python-unit red since 9f629685, not rechecked. Volt: `tsc -p tsconfig.app.json` on base `1c66de1ad9` exits 1 with 1122 errors (17 under agents); a PR must add no new line. `vitest` unit agents on that base: 69 files / 823 passed.

## 8. Open Linear tickets created in this session (all under AI-816)

- AI-1054, AI-1055 (gateway allowlist path, handshake verify token)
- AI-1056–AI-1063 (staging E2E F81–F89)
- AI-1064–AI-1069 (#871)
- AI-1070–AI-1076 (#873)
- AI-1077 (inbound queue rework)
- AI-1078–AI-1082 (#874)
- AI-1083–AI-1086 (#875)
- AI-1087–AI-1093 (#877, #872)
- AI-1105..AI-1145 (wave 9 exit). AI-1121 was publish docs#651 (merged `f2211fa4`; ticket state not rechecked). AI-1145 #897 M8 test gap.
- AI-1146..AI-1162 (wave 10 exit). Notable: AI-1155 (router spec vs code on `credential_rejected`), AI-1161 (Cursor attribution), AI-1162 ("Connections are not copied").

Closed this session: AI-958, AI-1048, AI-1049, AI-1052, AI-1053. Wave 9 Done: AI-1129 (#892), AI-1142 (#897). Wave 10 Done: AI-1062, AI-1120, AI-1057, AI-1058, AI-1083, AI-1069, AI-1056.
