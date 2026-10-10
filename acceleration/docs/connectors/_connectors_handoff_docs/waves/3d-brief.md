Wave 3d: WhatsApp on the channel bridge with the customer's own Meta app (T51, AI-879), the direct-call proxy (T44, AI-873) and token export (T45, AI-874).
Repo: /Users/kanat/Projects/stream/Vision-Agents (GetStream/Vision-Agents, Go router in acceleration/)   Base: accelerate at ad3fffd0 (v0.6.25 + 6 colleague commits)

You are one agent in a wave of PRs. An orchestrator routes your report; it reads only your final report, never your transcript. Your role, ticket id, slot and inputs follow after this brief.

## Hard rules

- No behaviour change when nothing is configured (no connectors, connections, bindings, provider apps, destinations, contact rows, channel_threads rows, episode_cards). Staging runs with connectors OFF. Prove it with a control test whose expected value comes from a probe on base, not from your head.
- Never run or enable: `router plugins migrate`, any data move, ROUTER_CONNECTORS_ENABLED on any shared env, T23, anything against staging or prod. Do not touch how `internal/channels` lines are served (that is T62's gated switch).
- Migrations are expand-only, in your slot only (slot+100 for a second file). No new column on a table v0.6.25 reads through bun: use a side table. If origin/accelerate's newest migration is at or after your slot, stop and report.
- Never merge, post review comments to GitHub, or edit Linear. Authors push their branch and open a draft PR, then stop; a mechanic waits for CI and writes the PR body.
- Work only in your worktree, your test database and your scratchpad folder `<scratchpad>/pr-w3d-<t>/`. Branch `connectors/<slug>` from the brief.
- Never commit secrets, `.env` files, tokens or `acceleration/internal/config/testing.yaml`.
- The repo is public: no GCP projects, clusters, secret prefixes or private infra repos in code, commits or PR bodies.

## Technical rules

- Read first: root AGENTS.md, `.claude/skills/go-testing/SKILL.md`, `.claude/skills/parallel-agents/SKILL.md`, and /private/tmp/claude-504/-Users-kanat-Projects-stream-Vision-Agents/a0abf0a3-74bd-4641-9e95-9b4aed318e30/scratchpad/wave3-common.md (first production caller, Huma + APIError + doc tags, hardcoded values need a source, PR body format).
- Test DB, before ANY test: `source <scratchpad>/w3c/testdb.sh <your worktree> model_router_test_w3d_<t>` (reviewers: `..._<t>_rv`). It repoints testing.yaml (which wins over env), fails closed, creates the DBs and exports ROUTER_POSTGRES_DSN and ROUTER_REDIS_ADDR=localhost:56379. Restore testing.yaml with `git checkout --` before committing. Integration: `-tags integration -p 1`; report skipped counts.
- Generated files are regenerated, never hand-merged: `cd acceleration && go run ./cmd/openapi`; `cd sdks/go && go generate .`; `cd sdks/js && npm install && npm run types && npm test`. Check with `.agents/skills/pr-review/scripts/generated.sh . origin/accelerate`. API change → an other-SDKs note at the end of `.claude/skills/sdk/SKILL.md` (merge=union on base).
- Must pass: `cd acceleration && gofmt -l . && go build ./... && go vet ./... && go vet -tags integration ./...`.
- Never rebase. Before a merge commit: `git diff --check` and grep for conflict markers. Fixers work detached and push with `git push origin HEAD:<branch>`.
- Put the ticket id (AI-xxx) in every commit message.
- Authors: run touched suites only; skip the full integration run (the reviewer runs it once). Near ~300k tokens hand off.

## Slots (2026-10-11, one hour apart; all reserved, expected unused)

T51 (AI-879) 20261011100000 · T44 (AI-873) 20261011110000 · T45 (AI-874) 20261011120000. Newest on base: 20261009150000_turn_playout_moments.sql (colleague #749 holds 20261009160000). Check again on every report.

## Known base failures

These fail on the base sha; do not fix or report them.
- internal/api TestSIPTrunksSuite — 2 tests (from #752)
- internal/agent TestTurnRecordingSuite — expects agent id "agent-1", setup sets "agent-test-<nanos>" (AI-938)
- flaky (passed on ad3fffd0, failed before): TestConnectionEventsSuite (`connection_events_test.go:438`), TestHarnessSuite/TestTheModelAskingAgainDoesNotReplaceTheCallersImages, TestAgentSuite (timing)
- internal/imagerouter TestImageRouterIntegrationSuite — fails if the env DB does not exist (testdb.sh creates it)
- TestQuotaSuite — shared local Redis across runs
- order-dependent: TestDataMoveSuite, TestAppConfigSuite (fresh DB), TestDisplaySuite/TestAFinishedLoginMarksTheReplyThatAskedForIt, TestSessionUpdateSuite/TestAUserRenamesASessionThatEnded, TestSessionVerbsSuite/…OnItsOwn, SetupSuite duplicate-key race, TestKnowledgeUrlsSuite, TestLLMSuite/TestStreamGeneratesAResponseIDWhenTheCallerHasNone, -race in session.Manager.Create, Python test_modalities.py

## Evidence rules

- Cite `file:line`, a command or a commit for every claim. Mark anything not checked `unverified`. Vendor facts (Meta, Slack SDKs) cite the page you opened.
- Every new rule gets a mutation check: break it, see a test fail, restore it. Every line a finding names gets a test that fails when that line is reverted.
- Build and test on the PR merged onto the current base, not only its own head.

## Findings format

`[Blocker|Should fix|Nit|Question] file:line — problem — fix`. Blocker and Should fix block the merge. Nit and Question become tickets; never fix them in the same PR.

## Report

Use your role's report format exactly, within its line budget. Logs and probes go to `<scratchpad>/pr-w3d-<t>/<role>-<round>.md`; give the path. Near ~300k tokens, write a handoff (done, left, open questions) there and stop with `HANDOFF: <path>`.
