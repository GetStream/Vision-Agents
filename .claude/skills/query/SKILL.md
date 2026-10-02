---
name: query
description: How list endpoints take filters and sorts, modelled on Stream's query syntax. Read before adding a filter, a sort, or a query endpoint.
---

# Query

One filter language for every list, the one getstream.io uses for users, channels,
messages and calls. A caller who has written one Stream query can write ours.

Paging is the `pagination` skill (`.claude/skills/pagination/SKILL.md`). Read it first:
this skill decides which rows, that one decides where the page starts.

## Shape

A resource that needs more than a couple of equality filters gets
`POST /v1/<resource>/query`, operationId `query<Resources>`:

```json
{
  "filter": {"state": "closed", "created_at": {"$gte": "2026-09-01T00:00:00Z"}},
  "sort": [{"field": "created_at", "direction": -1}],
  "limit": 25,
  "cursor": "…"
}
```

The answer is a page, `{items, has_more, next_cursor}`, as in pagination.

- POST because a filter is a tree, and a tree in a query string is unreadable. It still reads
  and nothing changes.
- A simple `GET` list with a couple of query parameters stays as it is. Add `query<Resources>`
  beside it when it outgrows them; don't add a tenth query parameter.
- `filter` is optional. An empty filter is every row the caller may see.

Stream calls the field `filter_conditions`. We call it `filter`; the grammar is the same.

## Filter grammar

A filter is a JSON object. Keys are field names or logical operators, and all keys at one
level are ANDed.

| Operator | Meaning | Example |
| --- | --- | --- |
| `$eq` | equal; a bare value is shorthand | `{"state": "closed"}` |
| `$in` | equal to any of a list | `{"agent": {"$in": ["a", "b"]}}` |
| `$gt` `$gte` `$lt` `$lte` | range, on numbers and RFC3339 times | `{"created_at": {"$lt": "…"}}` |
| `$exists` | the field is set or not | `{"ended_at": {"$exists": false}}` |
| `$contains` | array holds the value; object holds every key and value | `{"custom": {"$contains": {"team": "x"}}}` |
| `$autocomplete` | prefix match on a name | `{"title": {"$autocomplete": "sup"}}` |
| `$q` | full-text match | `{"title": {"$q": "billing refund"}}` |
| `$and` `$or` | a list of filters | `{"$or": [{"state": "running"}, {"agent": "x"}]}` |

`custom.<key>` is shorthand for `$contains` on the custom object:
`{"custom.team": "x"}` is `{"custom": {"$contains": {"team": "x"}}}`.

Not supported, and a 400: `$ne`, `$nin`, `$nor`, `$not`, regexes. Negative filters walk
every row that does not match instead of seeking to the ones that do. Rewrite them as a
positive `$in`. Stream allows `$ne` only on `id` and booleans; we allow `$ne` only on booleans,
where it is just `$eq` with the other value.

## Rules

- Each endpoint declares an allow-list: field → operators, and the column each field reads.
  Anything else is a 400 that names the field and the operator. Never pass a caller's field
  name into SQL; it is looked up in the allow-list and the column comes from there.
- Only allow what an index serves. If `(customer_id, agent, created_at, id)` exists, `agent`
  gets `$eq` and `$in`; a field with no index gets nothing until it has one. Custom fields
  get `$eq`/`$contains` only, which a GIN index on the JSONB column answers.
- `$and`/`$or` nest one level deep at most, and `$in` takes at most 100 values. Deeper trees
  are a 400; they are how a list becomes a table scan.
- Scope is not a filter. The handler ANDs the customer, and for end users the user, onto
  whatever arrived, as `SessionFilter.UserID` does. A caller cannot widen it by writing
  `$or`.
- Values are typed by the field: a time must parse as RFC3339, a number as a number, an
  enum as one of its values. A wrong type is a 400, not an empty page.

## Sort

`sort` is a list of `{"field", "direction"}`, direction `1` ascending or `-1` descending, as
Stream does it. Fields come from a sort allow-list, usually two or three indexed columns.

- Omitted, the endpoint's default applies, e.g. `created_at` descending.
- The server appends `id` in the direction of the last key, so the order is total.
- The cursor holds every sort key and a fingerprint of the sort. A cursor sent with a
  different sort is a 400. See pagination for why the keys are in the cursor.
- One index per allowed sort, behind the filters it is used with.

## Paging

As in pagination: `limit` + `cursor`, never offset. Stream Chat's queries page by offset
with a 1,000 row ceiling; Stream Video's `queryCalls` hands back a `next` pointer. We do what
Video does.

- `filter` and `sort` are resent with every page. The cursor is only a position.
- The next page is the filter ANDed with `(sort keys, id) < (cursor values)`, flipped to `>`
  for ascending keys.
- Fetch `limit + 1` for `has_more`. No counts.

## Go

Put the grammar in one package and reuse it; an endpoint only supplies its allow-lists.

- A short allow-list needs no parser. `querySessions` (`internal/api/session_query.go`)
  declares its filter as closed Huma structs, with one struct per operator (`Equals`,
  `TextMatch`), so the schema validates the request and documents it too. Combinations
  the schema cannot express, such as a text search with `project_id`, are refused in
  `sessionQueryOf`. The cursor carries the sort, so a cursor from one sort is refused by
  another. Start here, and move to a tree only when `$and`, `$or` or `$in` are needed.
- Parse the body into a tree once, validating against the allow-list as you go. The
  handler gets a typed tree or a `huma.Error400BadRequest`.
- Compile the tree to bun `Where` clauses with bound values. Field names never reach SQL
  as text.
- Where a list merges rows held in memory with rows in Postgres, like live sessions in
  `Manager.find`, the same tree needs a `Match(item)` too, so both halves filter and sort
  alike.
- Document the allowed fields, operators and sorts in the operation's description; that is
  what a caller reads.
- Tests follow the `go-testing` skill: one per allowed operator, one for a rejected field
  and a rejected operator, one for a nested `$or`, and one that pages a filtered, sorted
  list to the end without repeating or skipping a row.

## SDKs

Go first, as AGENTS.md says. The filter is a plain map in every SDK (`map[string]any`, a
dict, an object), which is how Stream's SDKs take `filter_conditions`. Sort is a typed list.
List methods follow the cursor the way `Items.Unwind` does.
