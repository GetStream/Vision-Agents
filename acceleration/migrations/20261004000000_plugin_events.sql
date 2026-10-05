-- +goose Up

-- plugin_events are the MCP events a config subscribes to on its plugins, and what the agent
-- does with each one. Empty subscribes to nothing.
ALTER TABLE agent_configs ADD COLUMN plugin_events JSONB NOT NULL DEFAULT '[]';

-- One subscription per config, login and declared event. The token is the callback's path,
-- so the server delivering to it is told apart from anyone who guessed the route, and the
-- secret is what it signs each delivery with.
CREATE TABLE agent_plugin_event_subscriptions (
    id             TEXT PRIMARY KEY,
    customer_id    TEXT NOT NULL,
    config_id      TEXT NOT NULL,
    plugin_id      TEXT NOT NULL,
    user_id        TEXT NOT NULL DEFAULT '',
    event          TEXT NOT NULL,
    arguments      JSONB NOT NULL DEFAULT '{}',
    key            TEXT NOT NULL,
    token          TEXT NOT NULL,
    secret         TEXT NOT NULL,
    remote_id      TEXT NOT NULL DEFAULT '',
    refresh_before TIMESTAMPTZ,
    status         TEXT NOT NULL,
    error          TEXT NOT NULL DEFAULT '',
    created_at     TIMESTAMPTZ NOT NULL,
    updated_at     TIMESTAMPTZ NOT NULL,
    deleted_at     TIMESTAMPTZ
);
CREATE UNIQUE INDEX agent_plugin_event_subscriptions_one_idx
    ON agent_plugin_event_subscriptions (config_id, plugin_id, user_id, key)
    WHERE deleted_at IS NULL;
CREATE UNIQUE INDEX agent_plugin_event_subscriptions_token_idx
    ON agent_plugin_event_subscriptions (token);

-- Every event delivered, so a retried delivery opens no second conversation.
CREATE TABLE agent_plugin_event_deliveries (
    subscription_id TEXT NOT NULL,
    event_id        TEXT NOT NULL,
    received_at     TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (subscription_id, event_id)
);

-- +goose Down
DROP TABLE agent_plugin_event_deliveries;
DROP TABLE agent_plugin_event_subscriptions;
ALTER TABLE agent_configs DROP COLUMN plugin_events;
