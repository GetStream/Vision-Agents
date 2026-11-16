---
name: pagination
description: How list endpoints page through results. Read before adding a list endpoint or a limit parameter.
---

# Pagination

Never use limit/offset. Lists take `limit` + `cursor`.

- Offset is slow: Postgres reads and discards every skipped row. A cursor is an indexed `WHERE`.
- Offset is wrong when data changes: an insert repeats a row on the next page, a delete skips one.

## Rules

- Response is `{items, has_more, next_cursor}`, like `SessionPage`. Never a bare array.
- Cursors are opaque base64url. Clients never build or parse them.
- Sort ends with `id` so the order is total. The cursor holds every sort key.
- Fetch `limit + 1` to set `has_more`. No counts.
- Filters are resent with every page; the cursor only holds a position.
- A bad cursor is a 400.
- Index the sort keys behind the filter, e.g. `(customer_id, updated_at DESC, id DESC)`.
- A list that filters and sorts takes a body instead of query parameters: see the `query` skill.

## Example: querySessions

Spec: `SessionQuery` carries `limit` and `cursor` in the body, and `SessionPage` is the 200.

Position, in `internal/store/models.go`:

```go
type SessionPosition struct {
	UpdatedAt time.Time `json:"u"`
	ID        string    `json:"id"`
	Rank      float32   `json:"r,omitempty"`
}
```

Store, in `QuerySessions`:

```go
if after := filter.Cursor; after != nil {
	query = query.Where("(updated_at, id) < (?, ?)", after.UpdatedAt, after.ID)
}
query = query.Limit(SessionLimit(filter.Limit) + 1)
```

Handler: `decodeCursor` and `encodeCursor` in `internal/api/cursor.go`; `page` trims the
extra row and says whether it was there. The session cursor also records which sort it
came from, so a cursor from one sort is refused by another.

Live sessions that nothing records get the same cut and sort in `Manager.find`, on their
`created`, which is truncated to microseconds (what Postgres keeps) so that a cursor from
one half does not repeat the other. A recorded live session is listed through its row.

A text search sorts by relevance, and its cursor adds the rank:
`(rank, updated_at, id) < (?, ?, ?)`.

Go SDK: list methods return the page and take a cursor; `Items.Unwind` follows it.

## Lists without a cursor yet

`listCalls`, `listSimulationRuns` and `getCallEvents` take a `limit` and no way to continue.
