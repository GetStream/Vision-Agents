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

## Chunk Size — Ask, Don't Pick

The decision to chunk a batched statement, and the chunk size if so, is **the user's call, not the agent's**.

Before writing chunking code, ask the user two questions:

1. **Should we chunk this at all?** Not every batch needs chunking — if the input is bounded small by request validation, one statement may be enough.
2. **If yes, what chunk size?** Surface what is known so the user can decide with context, then wait for an answer:
   - The hard ceilings of the target engine (e.g. PostgreSQL caps a single statement at 65,535 bind parameters — see References).
   - Any external guidance you can link.
   - The realistic input range, derived from request validators or known caller behavior.

Do not silently pick a chunk size from training data, gut feel, or consistency with another constant in the codebase. Local precedent is a consistency argument, not evidence — if the user explicitly tells you to match an existing constant for consistency, fine, but say so in the doc comment and keep it separate from the evidence-based reasoning.

Once the user picks a size, use the project's chunking helper if one exists (search for a chunking utility) and document the rationale in the doc comment on the chunk-size constant, with links to the sources cited. If the choice has not been benchmarked against the target fleet, say so explicitly so the next reader knows it was reasoned, not measured.

## Narrow Exceptions

- **Per-iteration error handling diverges.** If each row's failure must produce a distinct user-facing error or branch the control flow, per-row statements may be necessary. State this in a comment.
- **The operation genuinely streams** (e.g. cursoring through unbounded results) — but this is a different pattern, not a loop of independent writes.

"It is simpler to write" is not an exception.

## When Reviewing or Generating Code

Before writing a `for … { db.Exec(...) }` block: stop and check whether the same work fits a single batched statement plus a chunking helper. If yes, write the batched version. If chunking is needed, ask the user for the size — do not pick it.

## References

Batch / multi-row INSERT syntax:
- PostgreSQL — *SQL Commands · INSERT*: https://www.postgresql.org/docs/current/sql-insert.html (search "multirow VALUES")

Performance and limits relevant to chunk-size selection:
- PostgreSQL — *Frontend/Backend Protocol · Message Formats*: https://www.postgresql.org/docs/current/protocol-message-formats.html (search "Bind" — 16-bit parameter count → 65,535 hard cap per statement)
