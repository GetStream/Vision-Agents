-- +goose Up

-- connector_authorization_attempts is one interactive acquisition in flight: a consent, a
-- reconnect, later a step-up or an admin consent (architecture doc, «Add» item 5, on
-- connectors/planning). Each is sealed, expires, and is consumed once. It is the
-- prototype's table (20260929170000_connectors.sql and 20260929190000's kek_version on
-- codex/connector-support at cf62af0d) plus kind.
--
-- A data move does not carry these rows (store.dataTables): an attempt lives minutes, is
-- sealed under this deployment's key, and finishes at this deployment's callback.

CREATE TABLE connector_authorization_attempts (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    -- Connections are soft deleted, so the row this points at outlives the attempt.
    connection_id TEXT NOT NULL REFERENCES connector_connections (id),
    -- consent, reconnect, step_up or admin_consent (store.Attempt*). Checked by the store,
    -- not here, because step_up and admin_consent arrive with T27 and a CHECK would make
    -- each new kind a migration.
    kind TEXT NOT NULL,
    -- SHA-256 of the OAuth state, so the bearer value itself is never stored
    -- (store.AuthorizationStateHash). Unique, so a callback finds at most one attempt.
    state_hash TEXT NOT NULL UNIQUE,
    attempt_sealed BYTEA NOT NULL,
    -- Every attempt is sealed, and key versions start at 1 (auth.NewSealerWithKeyring).
    kek_version INTEGER NOT NULL CHECK (kek_version >= 1),
    expires_at TIMESTAMPTZ NOT NULL,
    consumed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Creating an attempt first deletes the customer's expired ones
-- (store.CreateConnectorAuthorizationAttempt), which this index finds without a scan.
CREATE INDEX connector_authorization_attempts_expiry_idx
    ON connector_authorization_attempts (customer_id, expires_at);

-- +goose Down
DROP TABLE connector_authorization_attempts;
