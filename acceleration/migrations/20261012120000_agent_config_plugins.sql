-- +goose Up

-- agent_plugins and user_plugins become one list, plugins, where an entry each end user
-- connects with their own account carries "user": true. A plugin named under both was the
-- app's, as it stays.
UPDATE agent_configs SET agent_plugins = agent_plugins || COALESCE((
    SELECT jsonb_agg(entry || '{"user": true}' ORDER BY position)
    FROM jsonb_array_elements(user_plugins) WITH ORDINALITY AS named(entry, position)
    WHERE NOT EXISTS (
        SELECT 1 FROM jsonb_array_elements(agent_plugins) AS app(entry)
        WHERE app.entry->>'name' = named.entry->>'name')
), '[]'::jsonb);
ALTER TABLE agent_configs DROP COLUMN user_plugins;
ALTER TABLE agent_configs RENAME COLUMN agent_plugins TO plugins;

-- +goose Down
ALTER TABLE agent_configs RENAME COLUMN plugins TO agent_plugins;
ALTER TABLE agent_configs ADD COLUMN user_plugins JSONB NOT NULL DEFAULT '[]';
UPDATE agent_configs SET
    user_plugins = COALESCE((
        SELECT jsonb_agg(entry - 'user' ORDER BY position)
        FROM jsonb_array_elements(agent_plugins) WITH ORDINALITY AS entries(entry, position)
        WHERE COALESCE((entry->>'user')::boolean, false)
    ), '[]'::jsonb),
    agent_plugins = COALESCE((
        SELECT jsonb_agg(entry ORDER BY position)
        FROM jsonb_array_elements(agent_plugins) WITH ORDINALITY AS entries(entry, position)
        WHERE NOT COALESCE((entry->>'user')::boolean, false)
    ), '[]'::jsonb);
