-- +goose Up
-- The Stream app a customer registered as its own, and the keys the router acts in it with.
-- A customer with no row acts wherever stream.fallback says; one with a row never does,
-- whatever state the row is in. Disconnecting deletes the keys and keeps the row, so the
-- app stays the customer's and is not quietly handed to the fallback.
--
-- Neither table is moved with a customer's data or watched for changes: a sealed secret
-- opens only under the keyring of the deployment that sealed it, and registering an app
-- is something done again on the deployment it moves to.
CREATE TABLE stream_apps (
    customer_id TEXT PRIMARY KEY,
    stream_app_pk BIGINT NOT NULL UNIQUE CHECK (stream_app_pk > 0),
    organization_id TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL CHECK (state IN ('connected', 'disconnected', 'blocked')),
    state_reason TEXT NOT NULL DEFAULT '',
    primary_key TEXT,
    revision BIGINT NOT NULL DEFAULT 1,
    allow_guests BOOLEAN NOT NULL DEFAULT false,
    checks JSONB,
    checked_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_by TEXT NOT NULL DEFAULT ''
);

CREATE TABLE stream_app_keys (
    api_key TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES stream_apps (customer_id) ON DELETE CASCADE,
    secret_sealed BYTEA NOT NULL,
    kek_version INTEGER NOT NULL,
    secret_last4 TEXT NOT NULL,
    key_created_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'rejected')),
    rejected_at TIMESTAMPTZ,
    rejected_reason TEXT NOT NULL DEFAULT '',
    verified_at TIMESTAMPTZ,
    last_webhook_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX stream_app_keys_customer ON stream_app_keys (customer_id);

-- +goose Down
DROP TABLE stream_app_keys;
DROP TABLE stream_apps;
