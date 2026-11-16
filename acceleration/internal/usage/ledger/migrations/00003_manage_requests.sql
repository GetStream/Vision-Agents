-- +goose Up
-- Every account write made under an Idempotency-Key, in the transaction that makes it: a
-- credit, a suspension, a restore or an override. The key is one space across them, so a key
-- is answered as its first request only for the same operation, organization and body.
-- Credits written before this table are found by their credit_entries.idempotency_key.
CREATE TABLE manage_requests (
    idempotency_key TEXT PRIMARY KEY CHECK (idempotency_key <> ''),
    operation TEXT NOT NULL CHECK (operation IN ('add_credit', 'suspend', 'restore', 'set_limits')),
    organization_id TEXT NOT NULL,
    -- Hex SHA-256 of what the request asks for, as the service reads it.
    request_sha256 TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- +goose Down
DROP TABLE manage_requests;
