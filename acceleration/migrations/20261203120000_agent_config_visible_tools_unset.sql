-- +goose Up

-- visible_tools tells a config that never set it from one that set it to show no tool: NULL
-- shows search and web_search, and an empty list shows none. Every empty list stored so far
-- was written for "not set", so it becomes NULL.
--
-- A router built before this one writes an empty list for a config that never set it, which
-- this one reads as showing no tool. Until it is replaced, a config it saves stops showing
-- search and web_search; setting visible_tools again restores them.
ALTER TABLE agent_configs
    ALTER COLUMN visible_tools DROP NOT NULL,
    ALTER COLUMN visible_tools DROP DEFAULT;
UPDATE agent_configs SET visible_tools = NULL WHERE visible_tools = '[]'::jsonb;

-- +goose Down

-- A router built before this one reads an empty list as "not set", so a config set to show
-- no tool shows search and web_search again.
UPDATE agent_configs SET visible_tools = '[]'::jsonb WHERE visible_tools IS NULL;
ALTER TABLE agent_configs
    ALTER COLUMN visible_tools SET DEFAULT '[]',
    ALTER COLUMN visible_tools SET NOT NULL;
