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
- Index the sort keys behind the filter, e.g. `(customer_id, created_at DESC, id DESC)`.

## Example: listSessions

Spec: the shared `Cursor` parameter, and `SessionPage` as the 200.

Position, in `internal/store/models.go`:

```go
type SessionPosition struct {
	CreatedAt time.Time `json:"t"`
	ID        string    `json:"id"`
	Rank      float32   `json:"r,omitempty"`
}
```

Store, in `QuerySessions`:

```go
if after := filter.Cursor; after != nil {
	query = query.Where("(created_at, id) < (?, ?)", after.CreatedAt, after.ID)
}
query = query.Limit(SessionLimit(filter.Limit) + 1)
```

Handler: `decodeCursor` and `encodeCursor` in `internal/api/cursor.go`; `page` trims the
extra row and says whether it was there.

Live sessions get the same cut and sort in `Manager.find`. Their `created` is truncated to
microseconds, which is what Postgres keeps, or a cursor from one half repeats the other.

For `searchSessions`, the cursor adds the rank: `(rank, created_at, id) < (?, ?, ?)`.

Go SDK: list methods return the page and take a cursor; `Items.Unwind` follows it.

## Lists without a cursor yet

`listCalls`, `listSimulationRuns` and `getCallEvents` take a `limit` and no way to continue.
