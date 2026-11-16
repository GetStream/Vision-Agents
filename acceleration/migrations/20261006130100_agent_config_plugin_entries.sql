-- +goose Up

-- Each entry in agent_plugins and user_plugins becomes an object naming its plugin, {name,
-- readonly, scopes, toolsets, tools}, carrying what plugin_options said for it, and
-- plugin_options goes. An option for a plugin named in neither list was for an app login
-- still to be made, so that plugin joins agent_plugins rather than losing its options.
UPDATE agent_configs SET agent_plugins = agent_plugins || COALESCE((
    SELECT jsonb_agg(option->'plugin')
    FROM jsonb_array_elements(plugin_options) AS option
    WHERE NOT agent_plugins @> jsonb_build_array(option->'plugin')
      AND NOT user_plugins @> jsonb_build_array(option->'plugin')
), '[]'::jsonb);

UPDATE agent_configs SET
    agent_plugins = COALESCE((
        SELECT jsonb_agg(jsonb_build_object('name', named.id) || COALESCE((
            SELECT option - 'plugin' FROM jsonb_array_elements(plugin_options) AS option
            WHERE option->>'plugin' = named.id LIMIT 1
        ), '{}'::jsonb) ORDER BY named.position)
        FROM jsonb_array_elements_text(agent_plugins) WITH ORDINALITY AS named(id, position)
    ), '[]'::jsonb),
    user_plugins = COALESCE((
        SELECT jsonb_agg(jsonb_build_object('name', named.id) || COALESCE((
            SELECT option - 'plugin' FROM jsonb_array_elements(plugin_options) AS option
            WHERE option->>'plugin' = named.id LIMIT 1
        ), '{}'::jsonb) ORDER BY named.position)
        FROM jsonb_array_elements_text(user_plugins) WITH ORDINALITY AS named(id, position)
    ), '[]'::jsonb);

ALTER TABLE agent_configs DROP COLUMN plugin_options;

-- +goose Down
ALTER TABLE agent_configs ADD COLUMN plugin_options JSONB NOT NULL DEFAULT '[]';

UPDATE agent_configs SET
    plugin_options = COALESCE((
        SELECT jsonb_agg(option) FROM (
            SELECT DISTINCT ON (entry->>'name')
                jsonb_build_object('plugin', entry->'name') || (entry - 'name') AS option
            FROM (
                SELECT entry, 1 AS list, position
                FROM jsonb_array_elements(agent_plugins) WITH ORDINALITY AS named(entry, position)
                UNION ALL
                SELECT entry, 2, position
                FROM jsonb_array_elements(user_plugins) WITH ORDINALITY AS named(entry, position)
            ) AS entries
            WHERE entry - 'name' <> '{}'::jsonb
            ORDER BY entry->>'name', list, position
        ) AS options
    ), '[]'::jsonb),
    agent_plugins = COALESCE((
        SELECT jsonb_agg(entry->'name' ORDER BY position)
        FROM jsonb_array_elements(agent_plugins) WITH ORDINALITY AS entries(entry, position)
    ), '[]'::jsonb),
    user_plugins = COALESCE((
        SELECT jsonb_agg(entry->'name' ORDER BY position)
        FROM jsonb_array_elements(user_plugins) WITH ORDINALITY AS entries(entry, position)
    ), '[]'::jsonb);
