---
name: migration-order
description: Use when a PR adds a migration under acceleration/migrations and is about to merge or was just rebased, when the router refuses to start with goose "missing migrations", or when `dev.py check-migrations` fails. Says how to move the migration to a free version and what to update with it.
---

# Migration order

goose runs strictly. A database that applied `accelerate` refuses any migration older than its
newest one: the router stops at start with "found N missing migrations before current version".
A PR whose migration is older than the newest one on `accelerate` still merges cleanly. goose only
notices when it applies to a database.

## Check and fix

1. `git fetch origin`. The check fetches nothing.
2. `uv run --no-sync dev.py check-migrations` (`--base` defaults to `origin/accelerate`). It fails
   when a migration the branch adds is not after the newest one on base, or shares a version, and
   suggests a free version for each.
3. `git mv` the file to the suggested name.
4. Grep the repo for the old version number and update every reference. Tests that pin versions
   are the usual ones, such as a migration suite's before/after constants: "after" is the new
   version, "before" is the newest version that precedes it.
5. Rerun the check until it passes.
6. Run the touched migration tests against their own database (see the `parallel-agents` skill).

## A local database that ran the old version

It holds a version that has no file now, and it misses the new one below its newest. goose
refuses it. Recreate that dev or test database. Never edit `goose_db_version` on a shared one.

## Ask the human, do not guess

- The migration already ran on staging or production.
- Its SQL must run before a migration that is now older than it.
- Two open PRs want the same slot.
- The rename moves it past another migration that touches the same table.
