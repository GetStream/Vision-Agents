Wave 3c: device `instructions` (AI-937), the thread bar for other conversation types (AI-936), BYO provider app (AI-906), Linq iMessage (T36) and Telnyx SMS (T53) on the channel bridge.
Repo: /Users/kanat/Projects/stream/Vision-Agents (GetStream/Vision-Agents, Go router in acceleration/)   Base: accelerate at 01f03694 (v0.6.24 + one plugins fix)

You are one agent in a wave of PRs. An orchestrator routes your report; it reads only your final report, never your transcript. Your role, ticket id, slot and inputs follow after this brief.

## Hard rules

- No behaviour change when nothing is configured (no connectors, connections, bindings, provider apps, destinations, contact rows, channel_threads rows, episode_cards). Staging runs with connectors OFF. Prove it with a control test that matches base. AI-937 is the one approved exception only if the orchestrator says so (Q1).
- Never run or enable: `router plugins migrate`, any data move, ROUTER_CONNECTORS_ENABLED on any shared env, anything against staging or prod.
- Migrations are expand-only, in your slot only (slot+100 for a second file). No new column on a table v0.6.24 reads with `SELECT t.*` through bun: use a side table. If origin/accelerate's newest migration is at or after your slot, stop and report.
- Never merge, post review comments to GitHub, or edit Linear. Authors push their branch and open a draft PR, then stop; a mechanic waits for CI and writes the PR body.
- Work only in your worktree, your test database and your scratchpad folder `<scratchpad>/pr-w3c-<t>/`. Branch `connectors/<slug>` from the brief.
- Never commit secrets, `.env` files, tokens or `acceleration/internal/config/testing.yaml`.
- The repo is public: no GCP projects, clusters, secret prefixes or private infra repos in code, commits or PR bodies.

## Technical rules

- Read first: root AGENTS.md, `.claude/skills/go-testing/SKILL.md`, `.claude/skills/parallel-agents/SKILL.md`, and /private/tmp/claude-504/-Users-kanat-Projects-stream-Vision-Agents/a0abf0a3-74bd-4641-9e95-9b4aed318e30/scratchpad/wave3-common.md (first production caller, Huma + APIError + doc tags, hardcoded values need a source, PR body format).
- Test DB, before ANY test: `source <scratchpad>/w3c/testdb.sh <your worktree> model_router_test_w3c_<t>` (reviewers: `..._<t>_rv`). It repoints testing.yaml (which wins over env), fails closed, creates the DBs and exports ROUTER_POSTGRES_DSN and ROUTER_REDIS_ADDR=localhost:56379. Restore testing.yaml with `git checkout --` before committing. Integration: `-tags integration -p 1`; report skipped counts.
- Generated files are regenerated, never hand-merged: `cd acceleration && go run ./cmd/openapi`; `cd sdks/go && go generate .`; `cd sdks/js && npm install && npm run types && npm test`. Check with `.agents/skills/pr-review/scripts/generated.sh . origin/accelerate`. API change → an other-SDKs note: `.claude/skills/sdk/changes/<ticket>.md` if that folder exists on origin/accelerate (#795), else the end of `.claude/skills/sdk/SKILL.md`.
- Must pass: `cd acceleration && gofmt -l . && go build ./... && go vet ./... && go vet -tags integration ./...`.
- Never rebase. Before a merge commit: `git diff --check` and grep for conflict markers. Fixers work detached and push with `git push origin HEAD:<branch>`.
- Put the ticket id (AI-xxx) in every commit message.

## Slots (2026-10-09, one hour apart; merge order)

AI-937 20261009100000 · AI-936 20261009110000 · AI-906 20261009120000 · T36 (AI-863) 20261009130000 · T53 (AI-881) 20261009140000. Newest on base: 20261008140000_episodes_close.sql. Check it again on every report.

## Known base failures

These fail on the base sha; do not fix or report them.
- internal/api TestSIPTrunksSuite — 2 tests (from #752)
- internal/agent TestTurnRecordingSuite — expects agent id "agent-1", setup sets "agent-test-<nanos>" (AI-938)
- internal/agent TestAgentSuite — timing, flaky
- internal/harness TestHarnessSuite/TestTheModelAskingAgainDoesNotReplaceTheCallersImages — flaky (failed once in the 3c unit run, passed 3/3 suite and 5/5 alone)
- internal/api TestConnectionEventsSuite — flaky (TestADeliverySigned…, TestADisconnectedConnection…; `connection_events_test.go:438`)
- internal/imagerouter TestImageRouterIntegrationSuite — fails if the env DB does not exist (testdb.sh creates it)
- TestQuotaSuite — shared local Redis across runs
- flaky or order-dependent: TestDataMoveSuite, TestAppConfigSuite (fresh DB), TestDisplaySuite/TestAFinishedLoginMarksTheReplyThatAskedForIt, TestSessionUpdateSuite/TestAUserRenamesASessionThatEnded, TestSessionVerbsSuite/…OnItsOwn, SetupSuite duplicate-key race on a fresh DB, TestKnowledgeUrlsSuite, TestLLMSuite/TestStreamGeneratesAResponseIDWhenTheCallerHasNone, the -race in session.Manager.Create, Python test_modalities.py

## Evidence rules

- Cite `file:line`, a command or a commit for every claim. Mark anything not checked `unverified`. Vendor facts (Linq, Telnyx headers and formats) cite the page you opened.
- Every new rule gets a mutation check: break it, see a test fail, restore it. Every line a finding names gets a test that fails when that line is reverted.
- Build and test on the PR merged onto the current base, not only its own head.
- Run touched suites only; the full integration run happens once, before the final review.

## Findings format

`[Blocker|Should fix|Nit|Question] file:line — problem — fix`. Blocker and Should fix block the merge. Nit and Question become tickets; never fix them in the same PR.

## Context limit

Near ~300k tokens, write a handoff (done, left, open questions) to your scratchpad folder and stop with `HANDOFF: <path>`.

## Report

Use your role's report format exactly, within its line budget. Logs and probes go to `<scratchpad>/pr-w3c-<t>/<role>-<round>.md`; give the path.
