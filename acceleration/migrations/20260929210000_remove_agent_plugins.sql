-- +goose Up

DROP TABLE agent_plugin_connections;
ALTER TABLE agent_configs DROP COLUMN plugins;

-- +goose Down

ALTER TABLE agent_configs ADD COLUMN plugins JSONB NOT NULL DEFAULT '[]';
CREATE TABLE agent_plugin_connections (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    config_id TEXT NOT NULL,
    plugin_id TEXT NOT NULL,
    instance_url TEXT NOT NULL DEFAULT '',
    access_token TEXT NOT NULL DEFAULT '',
    refresh_token TEXT NOT NULL DEFAULT '',
    expires_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'pending',
    oauth_state TEXT NOT NULL DEFAULT '',
    code_verifier TEXT NOT NULL DEFAULT '',
    client_id TEXT NOT NULL DEFAULT '',
    token_endpoint TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ
);
CREATE UNIQUE INDEX agent_plugin_connections_one_idx
    ON agent_plugin_connections (config_id, plugin_id)
    WHERE deleted_at IS NULL;
CREATE UNIQUE INDEX agent_plugin_connections_state_idx
    ON agent_plugin_connections (oauth_state)
    WHERE deleted_at IS NULL AND oauth_state <> '';
CREATE INDEX agent_plugin_connections_config_idx
    ON agent_plugin_connections (customer_id, config_id)
    WHERE deleted_at IS NULL;
