---
paths:
  - "**/*.go"
---

# Batching Repeated DB Operations

## The Rule

Never write a tight loop that issues one DB statement per element when the same work can be done with a single batched statement (multi-VALUES `INSERT`, `IN`-clause `SELECT`/`UPDATE`/`DELETE`, bulk `UPSERT`, `COPY`).

Applies whenever the loop body executes a database call against a bounded slice or map.

## Why

- **Round-trip cost.** Each statement is a separate network hop, which dominates per-statement cost for small statements.
- **Transaction duration.** When the loop runs inside an open transaction, every extra round trip extends the window where the tx holds row locks and is exposed to lock contention.
- **Observability noise.** N statements show up as N entries in statement statistics views, N spans in tracing, and N log lines — drowning the actual operation.

## Good vs Bad

```go
// Bad — one DB statement per element, N round trips per transaction
for _, r := range rows {
    if err := singleRowOp(ctx, r); err != nil {
        return err
    }
}
```

```go
// Good — one batched statement covers the whole input
if err := batchedOp(ctx, rows); err != nil {
    return err
}
```

`singleRowOp` / `batchedOp` are placeholders. The point is one round trip for the whole input, not how the SQL is assembled. Prefer the batched API the data-access library exposes. For bun (`github.com/uptrace/bun`):

- Multi-row `INSERT`: `db.NewInsert().Model(&rows).Exec(ctx)`
- Bulk `UPDATE`: `db.NewUpdate().Model(&rows).Bulk().Exec(ctx)`
- Multi-row `WHERE … IN`: `Where("id IN (?)", bun.In(values))`

Raw multi-VALUES via `ExecContext` is fine when the ORM form doesn't fit; build placeholders + args explicitly.

## Chunk Size — From Evidence

Not every batch needs chunking: if request validation bounds the input small, one statement is enough.

When it does need chunking, derive the size from evidence, not from gut feel or another constant in
the codebase:

- The hard ceiling of the target engine (PostgreSQL caps a single statement at 65,535 bind parameters — see References).
- The realistic input range, from request validators or known caller behavior.

Use the project's chunking helper if one exists. State the reasoning in the doc comment on the
chunk-size constant, citing the engine limit it stays under. If the size was reasoned, not
benchmarked, say so there and in the PR description, so a reviewer can challenge it.

## Narrow Exceptions

- **Per-iteration error handling diverges.** If each row's failure must produce a distinct user-facing error or branch the control flow, per-row statements may be necessary. State this in a comment.
- **The operation genuinely streams** (e.g. cursoring through unbounded results) — but this is a different pattern, not a loop of independent writes.

"It is simpler to write" is not an exception.

## When Reviewing or Generating Code

Before writing a `for … { db.Exec(...) }` block: stop and check whether the same work fits a single batched statement plus a chunking helper. If yes, write the batched version. If chunking is needed, size it as above.

## References

Batch / multi-row INSERT syntax:
- PostgreSQL — *SQL Commands · INSERT*: https://www.postgresql.org/docs/current/sql-insert.html (search "multirow VALUES")

Performance and limits relevant to chunk-size selection:
- PostgreSQL — *Frontend/Backend Protocol · Message Formats*: https://www.postgresql.org/docs/current/protocol-message-formats.html (search "Bind" — 16-bit parameter count → 65,535 hard cap per statement)
