-- +goose Up
-- Connectors replaced plugins: a config naming plugins, plugin_events or mcp_servers is
-- refused, and nothing reads the tables or the config columns that held them.
DROP TABLE agent_plugin_event_deliveries;
DROP TABLE agent_plugin_event_subscriptions;
DROP TABLE agent_plugin_clients;
DROP TABLE agent_plugin_connections;

ALTER TABLE agent_configs
    DROP COLUMN plugins,
    DROP COLUMN plugin_events,
    DROP COLUMN mcp_servers;

-- A move replaying its changes refuses one for a table an export no longer carries.
DELETE FROM data_changes WHERE table_name IN ('agent_plugin_connections', 'agent_plugin_clients');

-- +goose Down
-- The tables and columns come back empty: what they held is gone.
ALTER TABLE agent_configs
    ADD COLUMN plugins JSONB NOT NULL DEFAULT '[]',
    ADD COLUMN plugin_events JSONB NOT NULL DEFAULT '[]',
    ADD COLUMN mcp_servers JSONB NOT NULL DEFAULT '[]';

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
    deleted_at TIMESTAMPTZ,
    user_id TEXT NOT NULL DEFAULT ''
);
CREATE INDEX agent_plugin_connections_config_idx ON agent_plugin_connections (customer_id, config_id) WHERE deleted_at IS NULL;
CREATE UNIQUE INDEX agent_plugin_connections_one_idx ON agent_plugin_connections (config_id, plugin_id, user_id) WHERE deleted_at IS NULL;
CREATE UNIQUE INDEX agent_plugin_connections_state_idx ON agent_plugin_connections (oauth_state) WHERE deleted_at IS NULL AND oauth_state <> '';
CREATE TRIGGER data_changes_recorded AFTER INSERT OR UPDATE OR DELETE ON agent_plugin_connections
    FOR EACH ROW EXECUTE FUNCTION record_data_change('customer_id', '', '', 'id');

CREATE TABLE agent_plugin_clients (
    customer_id TEXT NOT NULL,
    config_id TEXT NOT NULL,
    plugin_id TEXT NOT NULL,
    client_id TEXT NOT NULL,
    secret_sealed BYTEA,
    kek_version INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (customer_id, config_id, plugin_id)
);
CREATE TRIGGER data_changes_recorded AFTER INSERT OR UPDATE OR DELETE ON agent_plugin_clients
    FOR EACH ROW EXECUTE FUNCTION record_data_change('customer_id', '', '', 'customer_id', 'config_id', 'plugin_id');

CREATE TABLE agent_plugin_event_subscriptions (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    config_id TEXT NOT NULL,
    plugin_id TEXT NOT NULL,
    user_id TEXT NOT NULL DEFAULT '',
    event TEXT NOT NULL,
    arguments JSONB NOT NULL DEFAULT '{}',
    key TEXT NOT NULL,
    token TEXT NOT NULL,
    secret TEXT NOT NULL,
    remote_id TEXT NOT NULL DEFAULT '',
    refresh_before TIMESTAMPTZ,
    status TEXT NOT NULL,
    error TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    deleted_at TIMESTAMPTZ
);
CREATE UNIQUE INDEX agent_plugin_event_subscriptions_one_idx ON agent_plugin_event_subscriptions (config_id, plugin_id, user_id, key) WHERE deleted_at IS NULL;
CREATE UNIQUE INDEX agent_plugin_event_subscriptions_token_idx ON agent_plugin_event_subscriptions (token);

CREATE TABLE agent_plugin_event_deliveries (
    subscription_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    received_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (subscription_id, event_id)
);
