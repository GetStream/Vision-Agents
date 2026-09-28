-- +goose Up

-- Agent grants name capabilities; reusable connections own the credentials. A connector
-- connection is tenant-scoped and can be bound to more than one agent config.
ALTER TABLE agent_configs ADD COLUMN connectors JSONB NOT NULL DEFAULT '[]';

CREATE TABLE connector_connections (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    connector_id TEXT NOT NULL,
    owner_type TEXT NOT NULL CHECK (owner_type IN ('app', 'user')),
    owner_id TEXT NOT NULL DEFAULT '',
    endpoint TEXT NOT NULL,
    instance TEXT NOT NULL DEFAULT '',
    label TEXT NOT NULL DEFAULT '',
    account_id TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    granted_scopes JSONB NOT NULL DEFAULT '[]',
    revision INTEGER NOT NULL DEFAULT 1,
    credential_sealed BYTEA NOT NULL DEFAULT ''::bytea,
    credential_kek_version INTEGER NOT NULL DEFAULT 1,
    expires_at TIMESTAMPTZ,
    cached_tools JSONB NOT NULL DEFAULT '[]',
    tools_digest TEXT NOT NULL DEFAULT '',
    tools_checked_at TIMESTAMPTZ,
    last_error TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ,
    CHECK ((owner_type = 'app' AND owner_id = '') OR (owner_type = 'user' AND owner_id <> ''))
);

CREATE INDEX connector_connections_owner_idx
    ON connector_connections (customer_id, owner_type, owner_id, connector_id)
    WHERE deleted_at IS NULL;

CREATE TABLE connector_authorization_attempts (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    connection_id TEXT NOT NULL REFERENCES connector_connections(id),
    state_hash TEXT NOT NULL UNIQUE,
    attempt_sealed BYTEA NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    consumed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX connector_authorization_attempts_expiry_idx
    ON connector_authorization_attempts (expires_at)
    WHERE consumed_at IS NULL;

-- +goose Down
DROP TABLE connector_authorization_attempts;
DROP TABLE connector_connections;
ALTER TABLE agent_configs DROP COLUMN connectors;
