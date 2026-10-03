-- +goose Up

-- Plugins each end user connects with their own account, from the conversation, beside the
-- ones the app connects once for every session.
ALTER TABLE agent_configs ADD COLUMN user_plugins JSONB NOT NULL DEFAULT '[]';

-- A connection made by an end user names them; the app's own names nobody. One live login
-- per config, plugin and user, so a user connecting again replaces their own attempt and
-- never somebody else's.
ALTER TABLE agent_plugin_connections ADD COLUMN user_id TEXT NOT NULL DEFAULT '';
DROP INDEX agent_plugin_connections_one_idx;
CREATE UNIQUE INDEX agent_plugin_connections_one_idx
    ON agent_plugin_connections (config_id, plugin_id, user_id)
    WHERE deleted_at IS NULL;

-- +goose Down
DROP INDEX agent_plugin_connections_one_idx;
DELETE FROM agent_plugin_connections WHERE user_id <> '';
CREATE UNIQUE INDEX agent_plugin_connections_one_idx
    ON agent_plugin_connections (config_id, plugin_id)
    WHERE deleted_at IS NULL;
ALTER TABLE agent_plugin_connections DROP COLUMN user_id;
ALTER TABLE agent_configs DROP COLUMN user_plugins;
