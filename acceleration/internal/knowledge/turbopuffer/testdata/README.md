# turbopuffer responses

These bodies are built from turbopuffer's API reference (read 2026-10-23), not recorded
from real calls: the fields, their names and their nesting follow the documented response
of each endpoint, and the values are made up to be easy to check.

- `query.json`: `POST /v2/namespaces/:namespace/query` (docs/query), a BM25 search on a
  namespace under the 1.28 GB minimum, so `billable_logical_bytes_queried` is the minimum.
- `upsert.json` and `delete.json`: `POST /v2/namespaces/:namespace` (docs/write), with
  `billable_logical_bytes_written` before the batch discount.
- `namespaces_1.json` and `namespaces_2.json`: `GET /v1/namespaces` (docs/namespaces), two
  pages joined by `next_cursor`.
- `metadata.json`: `GET /v1/namespaces/:namespace/metadata` (docs/metadata).

`TestEveryCallReportsWhatItBilled` makes the same calls against turbopuffer itself when
`TURBOPUFFER_API_KEY` is set.
