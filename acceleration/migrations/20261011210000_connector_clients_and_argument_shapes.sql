-- +goose Up

-- connector_connection_clients names the OAuth client a connection's grant was issued to
-- (AI-990 F16): its registration (customer, managed, operator, cimd or dcr) and client_id, so
-- a client registered on the fly (RFC 7591), such as Linear's, can be read without unsealing
-- the connection's credentials. A client_id is not a secret (RFC 6749 section 2.2); no secret
-- is stored here.
--
--   - A side table, not columns on connector_connections: the router before this one reads
--     connector_connections through bun.
--   - One row for a connection whose scheme names its client (core.ClientNamer), written by
--     each consent; none for a static scheme, or before the first consent.
--   - The row goes with its connection.
--
-- A data move does not carry these rows (store.dataTables): the next consent writes it again.
CREATE TABLE connector_connection_clients (
    connection_id TEXT PRIMARY KEY REFERENCES connector_connections (id) ON DELETE CASCADE,
    registration TEXT NOT NULL,
    client_id TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

-- connector_invocation_arguments is the shape of the arguments a connector tool call was
-- asked with (AI-990 F40): each top-level argument's name, its JSON type and, for a string or
-- an array, its length. Never a value, so the row still holds nothing the call was asked.
--
--   - A side table, not a column on connector_invocations: the router before this one reads
--     connector_invocations through bun.
--   - One row for a call whose arguments were a JSON object; none for one whose arguments did
--     not parse.
--   - The rows go with their invocation, and so with its connection.
--
-- A data move does not carry these rows, as it does not carry connector_invocations.
CREATE TABLE connector_invocation_arguments (
    invocation_id TEXT PRIMARY KEY REFERENCES connector_invocations (id) ON DELETE CASCADE,
    shape JSONB NOT NULL
);

-- +goose Down
DROP TABLE connector_invocation_arguments;
DROP TABLE connector_connection_clients;
