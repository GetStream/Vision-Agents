-- +goose Up

-- plugin_options change how a catalog plugin an agent names is reached and what its login
-- asks for: a list of {plugin, readonly, scopes}. Empty leaves every plugin the catalog's.
ALTER TABLE agent_configs ADD COLUMN plugin_options JSONB NOT NULL DEFAULT '[]';

-- +goose Down
ALTER TABLE agent_configs DROP COLUMN plugin_options;
