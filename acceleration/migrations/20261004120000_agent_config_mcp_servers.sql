-- +goose Up

-- mcp_servers are MCP servers outside the plugin catalog that an agent's sessions open
-- by their URL, with no login: a list of {name, url}. Empty is none.
ALTER TABLE agent_configs ADD COLUMN mcp_servers JSONB NOT NULL DEFAULT '[]';

-- +goose Down
ALTER TABLE agent_configs DROP COLUMN mcp_servers;
