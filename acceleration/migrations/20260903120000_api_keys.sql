-- +goose Up

-- Who the customer is, when the router is the one deciding rather than a proxy in front
-- of it. Rows elsewhere are keyed by an app id, so these two tables are what an app id
-- means: an app belongs to an organization, and a key belongs to an app.
CREATE TABLE organizations (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE apps (
    id TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL REFERENCES organizations (id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX apps_organization_idx ON apps (organization_id, created_at DESC);

-- The credential itself. The id is the public half: it travels in the clear, names which
-- secret to verify with, and is what a log line shows, because naming a key in a log is
-- how an operator revokes the right one at three in the morning.
CREATE TABLE api_keys (
    id TEXT PRIMARY KEY,
    app_id TEXT NOT NULL REFERENCES apps (id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    environment TEXT NOT NULL DEFAULT 'live',
    -- Verifying a token means recomputing its signature, which means holding the key
    -- material, so the secret cannot be hashed. It is sealed with AES-256-GCM under a key
    -- held outside the database; kek_version lets that key be rotated by re-wrapping rows
    -- rather than by reissuing every secret.
    secret_sealed BYTEA NOT NULL,
    kek_version INT NOT NULL DEFAULT 1,
    -- last4 is all of a secret the dashboard ever shows again.
    last4 TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by TEXT NOT NULL DEFAULT '',
    expires_at TIMESTAMPTZ,
    -- Written lazily: a synchronous update here would double the writes of a busy key.
    last_used_at TIMESTAMPTZ,
    -- A revoked key is kept rather than deleted, because a year of request rows
    -- referencing a key that no longer exists is unattributable noise.
    revoked_at TIMESTAMPTZ,
    revoked_by TEXT NOT NULL DEFAULT ''
);

CREATE INDEX api_keys_app_idx ON api_keys (app_id, created_at DESC);

-- +goose Down

DROP TABLE api_keys;
DROP TABLE apps;
DROP TABLE organizations;
