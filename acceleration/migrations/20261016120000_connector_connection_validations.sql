-- +goose Up

-- connector_connection_validations is the last validate of each connection (AI-1052): what it
-- found (a ConnectionValidationStatus), the code a program branches on or the provider's HTTP
-- status, why, for a person to read, and when. So a dashboard shows the last check after a
-- reload, and a failed one stays visible while the connection stays connected (a 5xx, a
-- timeout, a 429). The error is the one the validate answered with every value the credential
-- is sent as cut out, and capped at 1 KiB (api.storedError): a provider can echo a request in
-- its error (core AGENTS.md, «Secrets never print»).
--
--   - A side table, not columns on connector_connections: the router before this one reads
--     connector_connections through bun.
--   - One row per connection that was validated, replaced by each later validate; none before
--     the first.
--   - revision is the connector_connections.revision whose credentials the validate checked.
--     GET and list leave out a row behind the connection's revision: new credentials (a token
--     saved, a consent, a refresh) have not been validated.
--   - The row goes with its connection.
--
-- A data move does not carry these rows (store.dataTables): the next validate writes it again.
CREATE TABLE connector_connection_validations (
    connection_id TEXT PRIMARY KEY REFERENCES connector_connections (id) ON DELETE CASCADE,
    revision INTEGER NOT NULL,
    status TEXT NOT NULL,
    code TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    checked_at TIMESTAMPTZ NOT NULL
);

-- +goose Down
DROP TABLE connector_connection_validations;
