---
name: parallel-agents
description: Rules for running several coding agents on this repo at once (PR authors, reviewers, fixers, rebases), so they do not break each other's runs and do not burn tokens on repeated work. Read before you start more than one agent that writes code, runs integration tests or opens PRs.
---

# Parallel agents

These rules come from running seven connector PRs with agents on `accelerate`. Each one names
a mistake that actually happened.

## Keep agents out of each other's way

- **Test database.** `acceleration/internal/config/testing.yaml` wins over `ROUTER_POSTGRES_DSN`
  (`TestTheTestingFileWinsOverTheEnvironment`). Setting the env alone does not isolate a run.
  - Each agent points `testing.yaml` at a database of its own, such as `model_router_test_<slug>_test`.
    Do it through a `go test -overlay` copy or in the agent's own worktree.
  - It also sets `ROUTER_POSTGRES_DSN` (database `model_router_test_<slug>`) and `ROUTER_REDIS_ADDR`,
    because the `internal/api` suites skip silently without them.
  - It restores `testing.yaml` before it commits. Never commit it.
  - Two agents on `model_router_test` drop each other's schema in the middle of a run.
- **One Postgres server for all agents.** Agents share one test Postgres and each uses its own
  database on it. Do not start a Postgres container per agent. Eight of them, with fsync on, put
  the Docker VM into swap. Run the shared server with `fsync=off`, `synchronous_commit=off` and
  `full_page_writes=off`: the suites truncate their tables before every test, and each `TRUNCATE`
  waits on fsync.
  - Keep its data on disk, not on tmpfs. One server held 765 MB over 59 test databases.
  - Give each agent its own Redis. The router's Redis config has no database number or key
    prefix, so two agents on one Redis share counters, health and session keys.
  - When the agent is done, drop its databases (`DROP DATABASE … WITH (FORCE)`) and remove its
    Redis.
- **One package at a time.** Run integration tests with `-p 1`. Packages that run in parallel try to
  create the same new database and fail with a duplicate key.
- **Migration versions.** goose is strict. A migration older than the newest version a database
  has applied stops the router at start ("found N missing migrations before current version").
  - Give each PR its own version slot before the agents start, spaced apart. Never let two agents
    pick "the current time".
  - Merge in version order. Before each merge, check the newest file under `acceleration/migrations`
    on `origin/accelerate`.
  - A database that ran an earlier build of a PR can hold a version that a later merge lands below.
    Recreate that database.
- **Shared append files.** Every PR adds a note at the end of `.claude/skills/sdk/SKILL.md` and
  `CHANGELOG.md`, so any two open PRs conflict there. Keep both sides, the base's entries first.
- **Conflict markers.** Before every `git rebase --continue`, run `git diff --check` and grep for
  `<<<<<<<`, `=======` and `>>>>>>>`. Markers were committed twice by agents that skipped this.
- **Generated files.** Do not hand-merge `acceleration/api/openapi.yaml` or the generated clients.
  Take either side, regenerate them (see `acceleration/README.md`), and check the result with
  `.agents/skills/pr-review/scripts/generated.sh <repo root> origin/accelerate`.
- **CI runs the integration suites.** `Integration (router)` (`.github/workflows/integration.yml`)
  runs each Postgres package in its own job on every PR. Do not repeat those packages locally:
  run the suites you touched, and read CI for the rest.
  - CI leaves out the vendor suites that need API keys (tts, stt, llm, sts, search, imagegen,
    detection), and `imagerouter` and `dlc` skip themselves there. Run those by hand when you
    touch them, and list the runs in the PR.
  - The check is not required on `accelerate`. Wait for it to pass before you merge.

## Review and merge one PR at a time

Authors may write in parallel. Reviews and merges run one PR at a time.

- **Why.** Each merge makes the other open PRs conflict. When four PRs were reviewed at once, each
  review had to be redone on the merged tree, and some three or four times.
- **Order.** Merge one PR, then rebase the next one once onto the new base, then review it.
- **Review against the current base.** Build and test the PR merged onto `origin/accelerate`, not
  the PR head alone. Two PRs built cleanly on their own and failed to compile together.
- **Findings live in agent messages, not on GitHub.** Put the full list of earlier findings into
  every re-review prompt.
- **Re-review the delta.** The same reviewer reads
  `git range-diff <old base>..<old head> origin/accelerate..<new head>` and checks its own findings.
  A new finding in code the delta did not touch blocks the merge only if it is a Blocker.
- **A rebase after GO needs no new review** when the range-diff shows only `=` commits plus commits
  that change docs or tests. When Go code changed, the same reviewer checks the delta.
- **Nits go to a ticket**, never into the same PR. Each fix is new code, and new code brings new
  findings and another round.

## Spend fewer tokens

- **Fix with a fresh agent.** Give it the branch, the worktree and the list of findings. Do not
  resume the author: its context grew to 500k–700k tokens, and every tool call resends all of it.
- **Use a smaller model for mechanical work**: a rebase whose conflicts are only in docs, watching
  CI, editing a PR body.
- **Test only what changed while iterating.** Run the touched suites and the mutation checks. Do one
  full integration run before the final review, and the reviewer does not repeat it.
- **Ask the first reviewer to find everything.** Later rounds look only at the delta.
- **Wait for an agent's report.** Do not read its raw transcript file.
