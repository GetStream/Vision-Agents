---
name: pr
description: Create a draft pull request for the Vision-Agents repo using gh CLI.
---

# Pull Request (Vision-Agents)

## Before creating

- Run `git log main..HEAD --oneline`. If the branch contains more than one independent logical change, STOP and ask the user whether to split it before proceeding.
- Run `uv run --no-sync dev.py check`. Skip if the diff does not touch Python code (`*.py`) or `pyproject.toml` — e.g. docs-only, `.gitignore`, `.github/`, or `.claude/` changes.
- Do not run integration tests locally, CI handles them.
- If the change is user-facing (public API break, new feature, bug fix), update `CHANGELOG.md` per the rules in `CLAUDE.md`.

## Creating

- Always `gh pr create --draft`. Push the branch first.
- Follow `.github/pull_request_template.md`. Read every commit on the branch, do not summarise from the latest commit alone.

## Title

- Aim for 50 characters, never more than 65. The repo only squash-merges and uses the PR title as the commit subject on `accelerate`/`main` once a PR has two or more commits, and GitHub appends ` (#NNN)`. 65 plus that suffix stays within the 72 GitHub's docs give as the commit title maximum; 50 is what `git help commit` and Pro Git recommend.
- Say what changes, not how. The ticket id (`AI-833`) counts toward the limit; details go in the body.
- Bad, 85 characters: `feat(connectors): connector definitions table seeded from built-in manifests (AI-833)`. Good, 44: `AI-833: connector definitions with revisions`.

## Body

- Explain with pictures, not paragraphs. Near the top, a `## How it works` section shows the idea with a chart. Give small tables for "change → result → test that proves it" and for before/after.
  - **ASCII, in a fenced code block,** for branching logic (a tree) and data flow (boxes and arrows). Mermaid stacks these as tall columns of boxes; ASCII is shorter and reads the same in `gh pr view`. Draw with `├─ └─ │ ->` and keep each line within 80 columns, the default terminal width.
  - **A mermaid `sequenceDiagram`** for a protocol or concurrency. It spreads across, not down, so ASCII saves no height, and the actor columns leave too little width for the messages. No `;` inside a message (it splits statements).
  - Name each actor in full, or by a short form readers already know (`PG` for Postgres). No one-letter keys: the reader should not look anything up.
  - After creating the PR, open it and check that each chart renders and lines up.
- Keep prose short: one or two sentences per point, and nothing a diagram or table already shows. A reviewer should get the idea in a minute without a wall of text.
- Cite evidence: `file:line`, test names, commits. Mark anything not checked as unverified.
- `## Why` is motivation + context. `## Changes`, if included, is high-level; never per-bullet justifications, those belong in `## Why`.
- Link public GitHub issues inline within `## Why` (e.g. "users reported X (#478)"), not as a trailing `Fixes #N`.
- Do not paste CI, lint, or tool output in the body.
- Do not hard-wrap paragraphs. GitHub renders each newline inside a paragraph as a visible line break, so a 72-column-wrapped paragraph becomes a staircase. Write each paragraph or bullet as one unbroken line; rely on the browser to soft-wrap. Only use newlines to separate paragraphs, list items, or block elements.

## SQL changes

When a PR changes the shape of a query, for performance or any other reason, the body shows the evidence. Query plans are the one tool output the body carries.

- **Query, before and after, side by side.** Use an HTML table with one column per version. A markdown table cannot hold a code block.
- **Plans, before and after, collapsed.** Run `EXPLAIN (ANALYZE, BUFFERS)` on both versions against the same data, and put each in a `<details>` block. Compare buffers (`shared hit`, `read`), not only time: a warm cache can make a slow plan look fast. Say where you ran them and on how many rows.
- **The outcome, in one line or a small table.** Show what got better and by how much, both for the query (buffers, rows scanned, time) and for whoever uses it (the endpoint, the page, the job).

````markdown
<table>
<tr><th>Before</th><th>After</th></tr>
<tr><td>

```sql
SELECT ... ORDER BY revision DESC LIMIT 1
```

</td><td>

```sql
SELECT DISTINCT ON (customer_id, id) ...
```

</td></tr>
</table>

<details><summary>Plan before: <code>EXPLAIN (ANALYZE, BUFFERS)</code>, staging, 1.2M rows</summary>

```
Seq Scan on connector_definitions ... Buffers: shared hit=120 read=8410
```

</details>

<details><summary>Plan after</summary>

```
Index Scan using connector_definitions_pkey ... Buffers: shared hit=4
```

</details>

| | Before | After |
| --- | --- | --- |
| Buffers (hit + read) | 8,530 | 4 |
| Execution time | 412 ms | 0.08 ms |
| `GET /v1/agents/connectors` p95 | 450 ms | 15 ms |
````

The numbers in the example are made up, to show the shape. A real PR uses only numbers from plans it ran.
