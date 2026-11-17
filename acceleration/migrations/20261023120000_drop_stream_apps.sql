-- +goose Up
-- Customers' apps are acted in with the tokens the gateway mints, seeded for every app that
-- had registered by 20261022120000, so the registered keys are no longer read.
DROP TABLE stream_app_keys;
DROP TABLE stream_apps;

-- +goose Down
-- The tables come back empty: the keys they held are gone.
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
