-- +goose Up

-- agent_plugin_clients is the OAuth client an agent logs into one plugin with, for a plugin
-- such as Google Calendar that registers no client on the fly. It is set once for the agent,
-- and every login to the plugin made for that agent goes through it: the app's own, and
-- each end user's in a conversation. Two agents of the same app may use different clients.
--
-- The secret is sealed rather than hashed, because the token endpoint is sent it on every
-- exchange and refresh.
CREATE TABLE agent_plugin_clients (
    customer_id   TEXT NOT NULL,
    config_id     TEXT NOT NULL,
    plugin_id     TEXT NOT NULL,
    client_id     TEXT NOT NULL,
    secret_sealed BYTEA,
    kek_version   INTEGER NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ NOT NULL,
    updated_at    TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (customer_id, config_id, plugin_id)
);

-- +goose Down
DROP TABLE agent_plugin_clients;
